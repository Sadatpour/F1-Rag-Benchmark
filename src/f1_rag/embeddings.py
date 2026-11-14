from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from f1_rag.model_config import load_model_config, resolve_model_path


_MODEL_CONFIG = load_model_config()
EMBEDDING_MODEL_ID = _MODEL_CONFIG["embedding"]["repo_id"]
EMBEDDING_MODEL_PATH = resolve_model_path(_MODEL_CONFIG["embedding"]["local_path"])
EMBEDDING_WEIGHTS_PATH = resolve_model_path(_MODEL_CONFIG["embedding"]["weights_path"])


class MeanPoolingEncoder:
    def __init__(self, model: Any, tokenizer: Any, torch_module: Any, max_length: int = 256):
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch_module
        self.max_length = max_length
        self.device = torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        **_: Any,
    ) -> np.ndarray:
        vectors = []
        with self.torch.inference_mode():
            for start in range(0, len(sentences), batch_size):
                batch = self.tokenizer(
                    sentences[start : start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                batch = {key: value.to(self.device) for key, value in batch.items()}
                token_embeddings = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
                pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                if normalize_embeddings:
                    pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.append(pooled.cpu().numpy())
        if not vectors:
            return np.empty((0, int(self.model.config.hidden_size)), dtype=np.float32)
        return np.concatenate(vectors).astype(np.float32, copy=False)


def load_embedding_model(
    model_id: str = EMBEDDING_MODEL_ID,
    model_path: Path = EMBEDDING_MODEL_PATH,
    weights_path: Path = EMBEDDING_WEIGHTS_PATH,
) -> Any:
    if (model_path / "modules.json").is_file():
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Embedding dependencies are missing. Install them with "
                "`python -m pip install -r requirements-embedding.txt`."
            ) from error
        return SentenceTransformer(str(model_path))

    try:
        import torch
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Embedding dependencies are missing. Install them with "
            "`python -m pip install -r requirements-embedding.txt`."
        ) from error

    if weights_path.is_file():
        cache_dir = model_path / "hf-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        config = AutoConfig.from_pretrained(model_id, cache_dir=str(cache_dir))
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=str(cache_dir))
        model = AutoModel.from_config(config)
        state_dict = load_file(str(weights_path), device="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        unexpected_missing = [key for key in missing if not key.startswith("pooler.")]
        fixed_buffers = {"embeddings.position_ids", "embeddings.token_type_ids"}
        unexpected_keys = [
            key for key in unexpected if not key.startswith("pooler.") and key not in fixed_buffers
        ]
        if unexpected_missing or unexpected_keys:
            raise ValueError(
                f"Embedding weights do not match {model_id}; "
                f"missing keys: {unexpected_missing[:5]}, unexpected keys: {unexpected_keys[:5]}."
            )
        return MeanPoolingEncoder(model, tokenizer, torch)

    try:
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Embedding dependencies are missing. Install them with "
            "`python -m pip install -r requirements-embedding.txt`."
        ) from error

    if not (model_path / "modules.json").is_file():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=model_id, local_dir=str(model_path))
    return SentenceTransformer(str(model_path))
