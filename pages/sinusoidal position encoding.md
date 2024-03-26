- 来自经典论文attention is all you need，sinusoidal 对于 token 位置$$k=1, \cdots, L$$， 维度$d = 1, \cdots, d$
- $$p_{k, 2i} = \sin (k/10000^{2i/d}), p_{k, 2i+1} = \cos(k/10000^{2i/d})$$
- $L = 32, d = 128$时的sinusoidal positional encoding:
	- 黑色表示-1，白色表示1，灰色表示0
	- ![sinoidual-positional-encoding_1685374418562_0.png](../assets/sinoidual-positional-encoding_1685374418562_0_1711464554944_0.png){:height 269, :width 903}
- #加性位置编码
- ```python
    class SinusoidalPositionalEmbedding(nn.Embedding):
        """This module produces sinusoidal positional embeddings of any length."""
        def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None) -> None:
            super().__init__(num_positions, embedding_dim)
            self.weight = self._init_weight(self.weight)
  
        @staticmethod
        def _init_weight(out: nn.Parameter) -> nn.Parameter:
            """
            Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
            the 2nd half of the vector. [dim // 2:]
            """
            n_pos, dim = out.shape
            position_enc = np.array(
                [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
            )
            out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
            sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
            out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
            out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
            out.detach_()
            return out
  
        @torch.no_grad()
        def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0) -> torch.Tensor:
            """`input_ids_shape` is expected to be [bsz x seqlen]."""
            bsz, seq_len = input_ids_shape[:2]
            positions = torch.arange(
                past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
            )
            return super().forward(positions)
  ```