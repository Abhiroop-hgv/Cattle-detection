import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .config import Config


class EmbeddingNet(nn.Module):
    def __init__(self, backbone: str, embed_dim: int, pretrained: bool = True, img_size: int = 224):
        super().__init__()
        import timm

        try:  # ViTs need img_size to match our resize; position embeddings get resampled
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0,
                                              img_size=img_size)
        except TypeError:  # CNNs don't take img_size
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        # no bias: a shared bias term pushes every embedding toward the same direction
        self.head = nn.Linear(self.backbone.num_features, embed_dim, bias=False)
        self.data_config = timm.data.resolve_data_config({}, model=self.backbone)

    def forward(self, x):
        return F.normalize(self.head(self.backbone(x)), dim=-1)


class Embedder:
    """Image crops -> L2-normalised vectors."""

    def __init__(self, cfg: Config, device: str | None = None):
        import timm

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = EmbeddingNet(cfg.backbone, cfg.embed_dim, pretrained=cfg.checkpoint is None,
                                img_size=cfg.input_size)
        if cfg.checkpoint:
            self.net.load_state_dict(torch.load(cfg.checkpoint, map_location="cpu"))
        self.net.eval().to(self.device)
        self.transform = timm.data.create_transform(
            **{**self.net.data_config, "input_size": (3, cfg.input_size, cfg.input_size)}
        )

    @torch.no_grad()
    def embed(self, images: list[Image.Image]) -> np.ndarray:
        batch = torch.stack([self.transform(i.convert("RGB")) for i in images]).to(self.device)
        return self.net(batch).cpu().numpy().astype(np.float32)
