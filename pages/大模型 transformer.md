- |模型 | 位置编码|层归一化| 门控单元| Attention|  tokenizer| MoE|
  |---    |---------|--------| ----------|-----------|----|---|
  | [[LLAMA]] |RoPE| PreNorm| SwiGLU| | |
  | [[LLAMA 2]] |RoPE| PreNorm, RMSNorm| SwiGLU|GQA||no|
  | [[Mistral]] |RoPE| PreNorm| SwiGLU| GQA||no|
  | [[PaLM]] |RoPE|PreNorm| SwiGLU| MQA||no|
  |[[BLOOM]]|ALiBi|PreNorm，Embedding LayerNorm|GeLU, FFN||no|
  |[[GLM]]|RoPE|||GEGLU||no|
  | [[Grok]] |RoPE|PreNorm||MQA||yes||
  |[[Gemma]]|RoPE| SandwichNorm| GeGLU|MQA||no|
  |[[qwen]]|RoPE|PreNorm|SwiGLU|||no|
- [[位置编码]]
- [[层归一化]]
- [[transformer中的门控单元]]
- [[attention in fast way]]
- [[tokenizer]]