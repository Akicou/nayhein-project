from __future__ import annotations

from transformers import PretrainedConfig


class NayheinMiniConfig(PretrainedConfig):
    model_type = "nayhein_mini"

    def __init__(
        self,
        vocab_size: int = 32001,
        original_vocab_size: int | None = None,
        hidden_size: int = 512,
        num_hidden_layers: int = 10,
        num_attention_heads: int = 8,
        num_heads: int | None = None,
        head_dim: int = 64,
        intermediate_size: int = 2048,
        max_position_embeddings: int = 2048,
        mask_token_id: int | None = None,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        layer_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.original_vocab_size = original_vocab_size if original_vocab_size is not None else vocab_size - 1
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_heads = num_heads if num_heads is not None else num_attention_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.mask_token_id = mask_token_id if mask_token_id is not None else self.original_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.rope_theta = rope_theta
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
