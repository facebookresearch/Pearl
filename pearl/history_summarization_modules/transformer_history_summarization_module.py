# pyre-unsafe
import math
from typing import List

import torch
import torch.nn as nn
from pearl.api.action import Action
from pearl.api.observation import Observation
from pearl.history_summarization_modules.history_summarization_module import (
    HistorySummarizationModule,
)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard transformer sinusoidal positional encoding.
    """

    def __init__(self, d_model: int, max_len: int = 10_000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer so it moves with the module across devices and is saved in state_dict.
        self.register_buffer("pe", pe)  # [max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, d_model]
        """
        T = x.size(1)
        x = x + self.pe[:T, :].unsqueeze(0)
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learnable (absolute) positional encoding
    """

    def __init__(self, d_model: int, max_len: int = 10_000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embed = nn.Parameter(torch.zeros(max_len, d_model))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, d_model]
        """
        T = x.size(1)
        pos = self.pos_embed[:T, :].unsqueeze(0)  # [1, T, d_model]
        return self.dropout(x + pos)


def _generate_causal_mask(T: int, device: torch.device) -> torch.Tensor:
    """
    Returns a [T, T] causal (subsequent) mask where mask[i, j] = -inf if j>i (future),
    else 0.0. PyTorch Transformer expects additive mask with -inf where disallowed.
    """
    mask = torch.full((T, T), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


class TransformerHistorySummarizationModule(HistorySummarizationModule):
    """
    Decoder-style (causal) Transformer history summarizer.

    Maintains a sliding window of the last `history_length` (action, observation) pairs.
    At each step, encodes the full window with a causal mask and returns the last token's
    hidden state as the subjective state representation.
    """

    """
    Args:
        observation_dim: Dimension of observations.
        action_dim: Dimension of actions.
        history_length: Length of the history window (number of (action, observation) pairs).
        d_model: Dimension of the model.
        nhead: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        dim_feedforward: Dimension of the feedforward network.
        dropout: Dropout probability.
        layernorm_output: Whether to use layer normalization on the output.
        causal_masking: Whether to use causal masking.
        pos_encoding: Type of positional encoding ("sinusoidal" or "learned").
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        history_length: int = 8,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        layernorm_output: bool = True,
        causal_masking: bool = True,
        pos_encoding: str = "sinusoidal",
    ) -> None:
        super().__init__()
        self.history_length = history_length
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.input_dim = observation_dim + action_dim
        self.d_model = d_model
        self.causal_masking = causal_masking

        # Default tensors / FIFO history buffer
        self.register_buffer("default_action", torch.zeros((1, action_dim)))
        self.register_buffer(
            "history",
            torch.zeros((self.history_length, self.input_dim)),
        )

        # Input projection to model dimension
        self.input_proj = nn.Linear(self.input_dim, d_model)

        # Positional encoding
        if pos_encoding == "learned":
            self.pos_encoding = LearnedPositionalEncoding(
                d_model=d_model, dropout=dropout
            )
        elif pos_encoding == "sinusoidal":
            self.pos_encoding = SinusoidalPositionalEncoding(
                d_model=d_model, dropout=dropout
            )

        else:
            print("Invalid positional encoding type")
            exit(1)

        # Causal transformer encoder (decoder-only behavior via subsequent mask)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # so inputs/outputs are [B, T, C]
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model) if layernorm_output else nn.Identity()

    @torch.no_grad()
    def _append_history(self, obs_action: torch.Tensor) -> None:
        """
        obs_action: [1, input_dim]
        """
        self.history = torch.cat(
            [
                self.history[1:, :],
                obs_action.view(1, self.input_dim),
            ],
            dim=0,
        )

    def _encode_last_token(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: [B, T, input_dim]
        returns: [B, d_model] (last token)
        """
        device = next(self.parameters()).device
        x = self.input_proj(x_seq.to(device))  # [B, T, d_model]
        x = self.pos_encoding(x)  # [B, T, d_model]
        T = x.size(1)
        mask = (
            _generate_causal_mask(T, device=device)
            if self.causal_masking == True
            else None
        )  # [T, T]
        out = self.transformer(x, mask=mask)  # [B, T, d_model]
        return self.out_norm(out[:, -1, :])  # [B, d_model]

    def summarize_history(
        self, observation: Observation, action: Action | None
    ) -> torch.Tensor:
        """
        Returns the *last token* representation of the causal transformer over the
        window of history (including current (action, obs)).
        Output shape: [d_model]
        """
        assert isinstance(observation, torch.Tensor)
        obs = observation.clone().detach().float().view(1, self.observation_dim)

        if action is None:
            act = self.default_action
        else:
            assert isinstance(action, torch.Tensor)
            act = action.clone().detach().float().view(1, self.action_dim)

        # Concatenate (action, observation) for current step
        pair = torch.cat((act, obs), dim=-1)  # [1, input_dim]
        assert pair.shape[-1] == self.history.shape[-1]

        # Maintain FIFO buffer
        self._append_history(pair)

        # Prepare model inputs
        device = next(self.parameters()).device
        hist = self.history.to(device).unsqueeze(0)  # [1, T, input_dim]
        seq = self._encode_last_token(hist)
        return seq.squeeze(0)

    def get_history(self) -> torch.Tensor:
        return self.history

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, input_dim] — batch of sequences of (action, observation) pairs.
        Returns: [B, d_model] — representation of the last token for each sequence.
        """
        assert x.dim() == 3 and x.size(-1) == self.input_dim, (
            f"Expected [B, T, {self.input_dim}], got {tuple(x.shape)}"
        )

        return self._encode_last_token(x)

    def reset(self) -> None:
        self.history.zero_()

    def compare(self, other: HistorySummarizationModule) -> str:
        """
        Compares two HistorySummarizationModule instances for equality.

        Args:
        other: The other HistorySummarizationModule to compare with.

        Returns:
        str: A string describing the differences, or an empty string if they are identical.
        """

        differences: List[str] = []

        if not isinstance(other, TransformerHistorySummarizationModule):
            differences.append(
                "other is not an instance of TransformerHistorySummarizationModule"
            )
            return "\n".join(differences)

        # Basic scalar attrs
        if self.history_length != other.history_length:
            differences.append(
                f"history_length is different: {self.history_length} vs {other.history_length}"
            )
        if getattr(self, "filled_len", None) != getattr(other, "filled_len", None):
            differences.append(
                f"filled_len is different: "
                f"{getattr(self, 'filled_len', None)} vs "
                f"{getattr(other, 'filled_len', None)}"
            )
        if self.observation_dim != other.observation_dim:
            differences.append(
                f"observation_dim is different: {self.observation_dim} vs {other.observation_dim}"
            )
        if self.action_dim != other.action_dim:
            differences.append(
                f"action_dim is different: {self.action_dim} vs {other.action_dim}"
            )
        if self.d_model != other.d_model:
            differences.append(
                f"d_model is different: {self.d_model} vs {other.d_model}"
            )

        # Buffers
        if not torch.allclose(self.default_action, other.default_action):
            differences.append(
                f"default_action is different: {self.default_action} vs {other.default_action}"
            )
        if not torch.allclose(self.history, other.history):
            differences.append("history buffer is different")

        # Positional encoding (buffer + dropout p)
        if hasattr(self.pos_encoding, "pe") and hasattr(other.pos_encoding, "pe"):
            if not torch.allclose(self.pos_encoding.pe, other.pos_encoding.pe):
                differences.append("positional_encoding.pe is different")
        p_self = getattr(self.pos_encoding.dropout, "p", None)
        p_other = getattr(other.pos_encoding.dropout, "p", None)
        if p_self != p_other:
            differences.append(
                f"positional_encoding.dropout.p differs: {p_self} vs {p_other}"
            )

        return "\n".join(differences)

    def __repr__(self) -> str:
        """
        decrease verbosity of print
        """
        return (
            f"TransformerHistorySummarizationModule("
            f"d_model={self.d_model}, "
            f"layers={len(self.transformer.layers)})"
        )
