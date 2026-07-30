"""
MultiBanFakeDetect model.

Text encoder  : BanglaBERT-Large (Electra-based, 1024-d)
Image encoder : ViT-B/16 (768-d)
Projection    : W_t linear layer (1024 -> 768) to align dimensions
Fusion        : Cross-Modal Attention Fusion (CMAF)
                - bidirectional multi-head cross-attention
                - learned gate blending text+image into joint representation
Head          : 3-way classifier over {real, human_fake, llm_fake}
Explainability: ForwardWrapper for Captum IntegratedGradients

FIXES vs v1:
- Gate residual fixed: now blends text+image pooled (not just text)
- _freeze_text_layers: handles Electra encoder structure (BanglaBERT-Large)
  with graceful fallback so it never silently fails
- USE_FALLBACK references kept for backward compatibility
"""
import os
import sys

import timm
import torch
import torch.nn as nn
from transformers import AutoModel

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs import config as cfg


class CrossModalAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention between text and image token sequences.

    text_tokens  (B, T, D): text token sequence from BanglaBERT projection
    image_tokens (B, N, D): ViT patch token sequence

    Returns fused (B, D), text_pooled (B, D), image_pooled (B, D)
    """

    def __init__(self, dim: int = cfg.FUSION_HIDDEN_DIM,
                 heads: int = cfg.FUSION_HEADS,
                 dropout: float = cfg.DROPOUT):
        super().__init__()
        self.text_to_image_attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True)
        self.image_to_text_attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True)

        self.text_norm  = nn.LayerNorm(dim)
        self.image_norm = nn.LayerNorm(dim)

        # Gate: decides how much of the fused candidate to use
        # vs. averaging the two pooled representations
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )
        self.fuse_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, text_tokens, image_tokens, text_key_padding_mask=None):
        # Text attends to image
        text_attended, _ = self.text_to_image_attn(
            query=text_tokens, key=image_tokens, value=image_tokens,
        )
        text_attended = self.text_norm(text_attended + text_tokens)

        # Image attends to text
        image_attended, _ = self.image_to_text_attn(
            query=image_tokens, key=text_tokens, value=text_tokens,
            key_padding_mask=text_key_padding_mask,
        )
        image_attended = self.image_norm(image_attended + image_tokens)

        # Pool text (masked mean over non-padded tokens)
        if text_key_padding_mask is not None:
            mask = (~text_key_padding_mask).unsqueeze(-1).float()
            text_pooled = (text_attended * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        else:
            text_pooled = text_attended.mean(1)

        # Pool image (mean over all patch tokens)
        image_pooled = image_attended.mean(1)

        # Gated fusion
        # FIX v2: residual blends BOTH text and image pooled (not just text)
        concat          = torch.cat([text_pooled, image_pooled], dim=-1)  # (B, 2D)
        gate_val        = self.gate(concat)                               # (B, D)
        fused_candidate = self.fuse_proj(concat)                          # (B, D)
        residual        = (text_pooled + image_pooled) / 2               # (B, D)
        fused           = gate_val * fused_candidate + (1 - gate_val) * residual

        return fused, text_pooled, image_pooled


class MultiBanFakeDetectModel(nn.Module):

    def __init__(self, text_model_name: str = None, image_model_name: str = None,
                 num_classes: int = cfg.NUM_CLASSES,
                 freeze_text_layers: int = 0,
                 freeze_image: bool = False):
        super().__init__()

        text_model_name  = text_model_name  or (
            cfg.FALLBACK_TEXT_MODEL_NAME if cfg.USE_FALLBACK else cfg.TEXT_MODEL_NAME)
        image_model_name = image_model_name or (
            cfg.FALLBACK_IMAGE_MODEL_NAME if cfg.USE_FALLBACK else cfg.IMAGE_MODEL_NAME)

        # Text encoder (BanglaBERT-Large = Electra discriminator backbone)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_hidden       = self.text_encoder.config.hidden_size  # 1024

        # Image encoder (ViT-B/16)
        self.image_encoder = timm.create_model(image_model_name, pretrained=True, num_classes=0)
        image_hidden = (self.image_encoder.embed_dim
                        if hasattr(self.image_encoder, "embed_dim")
                        else cfg.IMAGE_HIDDEN_DIM)

        # Projection: align text 1024-d -> 768-d
        self.text_projection = nn.Linear(text_hidden, cfg.PROJECTION_DIM)
        self.image_projection = (
            nn.Identity() if image_hidden == cfg.PROJECTION_DIM
            else nn.Linear(image_hidden, cfg.PROJECTION_DIM)
        )

        self.fusion = CrossModalAttentionFusion(dim=cfg.PROJECTION_DIM)

        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.PROJECTION_DIM),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.PROJECTION_DIM, cfg.PROJECTION_DIM // 2),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.PROJECTION_DIM // 2, num_classes),
        )

        if freeze_text_layers > 0:
            self._freeze_text_layers(freeze_text_layers)
        if freeze_image:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

    def _freeze_text_layers(self, n_layers: int):
        """
        Freeze first n_layers of the text encoder.
        BanglaBERT-Large is Electra-based. Its encoder layers live at:
          self.text_encoder.encoder.layer   (standard BERT/Electra path)
        We also freeze embeddings. Both wrapped in try/except so a
        different architecture never silently corrupts the freeze.
        """
        try:
            for p in self.text_encoder.embeddings.parameters():
                p.requires_grad = False
        except AttributeError:
            print("Warning: could not freeze embeddings (unexpected model structure)")

        try:
            layers = self.text_encoder.encoder.layer
            n = min(n_layers, len(layers))
            for i in range(n):
                for p in layers[i].parameters():
                    p.requires_grad = False
            print(f"Froze {n}/{len(layers)} encoder layers + embeddings")
        except AttributeError:
            print("Warning: could not freeze encoder layers (unexpected model structure)")

    def _image_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.image_encoder.forward_features(pixel_values)
        if feats.dim() == 3:
            return feats          # (B, N, D) — standard ViT patch token output
        return feats.unsqueeze(1) # fallback for pooled backbones

    def forward(self, input_ids, attention_mask, pixel_values,
                return_embeddings: bool = False):
        text_out    = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_tokens = self.text_projection(text_out.last_hidden_state)  # (B, T, 768)

        image_tokens = self._image_tokens(pixel_values)
        image_tokens = self.image_projection(image_tokens)              # (B, N, 768)

        key_padding_mask = (attention_mask == 0)  # True where padded

        fused, text_pooled, image_pooled = self.fusion(
            text_tokens, image_tokens,
            text_key_padding_mask=key_padding_mask,
        )
        logits = self.classifier(fused)

        if return_embeddings:
            return logits, {
                "fused":        fused,
                "text_pooled":  text_pooled,
                "image_pooled": image_pooled,
            }
        return logits


class ForwardWrapper(nn.Module):
    """
    Captum-compatible wrapper for IntegratedGradients.

    CRITICAL argument order: (inputs_embeds, pixel_values, attention_mask)
    Captum calls forward_func(*inputs, *additional_forward_args), so:
      inputs                 = (inputs_embeds, pixel_values)
      additional_forward_args = (attention_mask,)
    Results in call: wrapper(inputs_embeds, pixel_values, attention_mask)
    Getting this wrong silently swaps pixel_values and attention_mask.
    """

    def __init__(self, model: MultiBanFakeDetectModel):
        super().__init__()
        self.model = model

    def forward(self, inputs_embeds, pixel_values, attention_mask):
        text_out     = self.model.text_encoder(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        text_tokens  = self.model.text_projection(text_out.last_hidden_state)

        image_tokens = self.model._image_tokens(pixel_values)
        image_tokens = self.model.image_projection(image_tokens)

        key_padding_mask = (attention_mask == 0)
        fused, _, _      = self.model.fusion(
            text_tokens, image_tokens,
            text_key_padding_mask=key_padding_mask,
        )
        return self.model.classifier(fused)
