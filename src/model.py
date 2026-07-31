"""
MultiBanFakeDetect model — FINAL v3

Text:  BanglaBERT-Large (Electra, 1024-d)
Image: ViT-B/16 (768-d, 197 patch tokens)
Proj:  W_t linear 1024→768
Fuse:  CMAF — bidirectional cross-attention + learned gate
Head:  3-way softmax {real, human_fake, llm_fake}
XAI:   ForwardWrapper for Captum IntegratedGradients

BUGS FIXED vs v1/v2:
1. Gate residual now averages (text+image)/2 — was text-only, dropped image when gate→0
2. _freeze_text_layers: try/except for Electra structure (BanglaBERT-Large)
3. ForwardWrapper arg order: (embeds, pixels, mask) — Captum calls *inputs, *additional_args
"""
import os, sys
import timm
import torch
import torch.nn as nn
from transformers import AutoModel

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs import config as cfg


class CrossModalAttentionFusion(nn.Module):
    def __init__(self, dim=cfg.FUSION_HIDDEN_DIM, heads=cfg.FUSION_HEADS, dropout=cfg.DROPOUT):
        super().__init__()
        self.t2i = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.i2t = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.t_norm = nn.LayerNorm(dim)
        self.i_norm = nn.LayerNorm(dim)
        self.gate = nn.Sequential(nn.Linear(dim*2, dim), nn.Sigmoid())
        self.fuse = nn.Sequential(nn.Linear(dim*2, dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, text_tokens, image_tokens, text_key_padding_mask=None):
        ta, _ = self.t2i(text_tokens,  image_tokens, image_tokens)
        ta     = self.t_norm(ta + text_tokens)
        ia, _ = self.i2t(image_tokens, text_tokens,  text_tokens,
                          key_padding_mask=text_key_padding_mask)
        ia     = self.i_norm(ia + image_tokens)

        if text_key_padding_mask is not None:
            mask = (~text_key_padding_mask).unsqueeze(-1).float()
            tp   = (ta * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        else:
            tp = ta.mean(1)
        ip = ia.mean(1)

        cat   = torch.cat([tp, ip], dim=-1)
        g     = self.gate(cat)
        fused = g * self.fuse(cat) + (1 - g) * (tp + ip) / 2  # BUG FIX: both modalities in residual
        return fused, tp, ip


class MultiBanFakeDetectModel(nn.Module):
    def __init__(self, text_model_name=None, image_model_name=None,
                 num_classes=cfg.NUM_CLASSES,
                 freeze_text_layers=0, freeze_image=False):
        super().__init__()
        text_model_name  = text_model_name  or (cfg.FALLBACK_TEXT_MODEL_NAME  if cfg.USE_FALLBACK else cfg.TEXT_MODEL_NAME)
        image_model_name = image_model_name or (cfg.FALLBACK_IMAGE_MODEL_NAME if cfg.USE_FALLBACK else cfg.IMAGE_MODEL_NAME)

        self.text_encoder  = AutoModel.from_pretrained(text_model_name)
        text_hidden        = self.text_encoder.config.hidden_size

        self.image_encoder = timm.create_model(image_model_name, pretrained=True, num_classes=0)
        image_hidden       = getattr(self.image_encoder, 'embed_dim', cfg.IMAGE_HIDDEN_DIM)

        self.text_projection  = nn.Linear(text_hidden, cfg.PROJECTION_DIM)
        self.image_projection = nn.Identity() if image_hidden == cfg.PROJECTION_DIM \
                                else nn.Linear(image_hidden, cfg.PROJECTION_DIM)

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

    def _freeze_text_layers(self, n):
        try:
            for p in self.text_encoder.embeddings.parameters():
                p.requires_grad = False
        except AttributeError:
            print("Warning: could not freeze embeddings")
        try:
            layers = self.text_encoder.encoder.layer
            for i in range(min(n, len(layers))):
                for p in layers[i].parameters():
                    p.requires_grad = False
            print(f"Froze {min(n,len(layers))}/{len(layers)} encoder layers")
        except AttributeError:
            print("Warning: could not freeze encoder layers")

    def _image_tokens(self, pixels):
        f = self.image_encoder.forward_features(pixels)
        return f if f.dim() == 3 else f.unsqueeze(1)

    def forward(self, input_ids, attention_mask, pixel_values, return_embeddings=False):
        t_out    = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        t_tokens = self.text_projection(t_out.last_hidden_state)
        i_tokens = self.image_projection(self._image_tokens(pixel_values))
        kpm      = (attention_mask == 0)
        fused, tp, ip = self.fusion(t_tokens, i_tokens, text_key_padding_mask=kpm)
        logits   = self.classifier(fused)
        if return_embeddings:
            return logits, {"fused": fused, "text_pooled": tp, "image_pooled": ip}
        return logits


class ForwardWrapper(nn.Module):
    """
    Captum IG wrapper. CRITICAL arg order: (embeds, pixels, mask)
    Captum calls: wrapper(*inputs, *additional_forward_args)
    where inputs=(embeds, pixels) and additional=(mask,)
    → wrapper(embeds, pixels, mask)  ← must match this exactly
    """
    def __init__(self, model: MultiBanFakeDetectModel):
        super().__init__()
        self.model = model

    def forward(self, inputs_embeds, pixel_values, attention_mask):
        t_out    = self.model.text_encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        t_tokens = self.model.text_projection(t_out.last_hidden_state)
        i_tokens = self.model.image_projection(self.model._image_tokens(pixel_values))
        kpm      = (attention_mask == 0)
        fused, _, _ = self.model.fusion(t_tokens, i_tokens, text_key_padding_mask=kpm)
        return self.model.classifier(fused)
