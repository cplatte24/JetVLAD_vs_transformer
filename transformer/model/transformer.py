import torch
import torch.nn as nn

class TransformerTagger(nn.Module):
    """Jet tagger operating on padded track sequences.

    Inputs
    ------
    x    : float tensor [B, N, F]  z-scored per-track features, zero-padded
    mask : bool tensor  [B, N]     True where the entry is PADDING
                                   (torch key_padding_mask convention)

    Output
    ------
    logits : float tensor [B, n_classes]
    """

    def __init__(self, feature_size, d_model=64, n_heads=8, n_layers=4,
                 dim_feedforward=None, dropout=0.1, n_classes=3):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        # Per-track embedding MLP (kept from the NetVLAD design, then
        # projected to d_model).
        self.embed = nn.Sequential(
            nn.Linear(feature_size, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # Learnable class token used for pooling.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,       # pre-norm: much more stable on small data
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x, mask):
        B = x.size(0)

        # Zero out padded rows *before* the embedding so garbage values can
        # never propagate (belt and braces on top of the attention mask).
        x = x.masked_fill(mask.unsqueeze(-1), 0.0)

        h = self.embed(x)                                     # [B, N, D]

        cls = self.cls_token.expand(B, -1, -1)                # [B, 1, D]
        h = torch.cat((cls, h), dim=1)                        # [B, 1+N, D]

        # The class token is never padding.
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
        full_mask = torch.cat((cls_mask, mask), dim=1)        # [B, 1+N]

        h = self.encoder(h, src_key_padding_mask=full_mask)   # [B, 1+N, D]
        h = self.final_norm(h[:, 0])                          # class token

        return self.head(h)
