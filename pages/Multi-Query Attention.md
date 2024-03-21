- 多查询注意力(Multi Query Attention)是多头注意力的一种变体。其主要区别在于，在多查询注意力中不同的注意力头共享一个键和值的集合，每个头只单独保留了一份查询参数。因此键和值的矩阵仅有一份，这大幅度减少了显存占用，使其更高效。
	- 由于多查询注意力改变了注意力机制的结构，因此模型通常需要从训练开始就支持多查询注意力。包括 [[Falcon]], [[StarCoder]]等在内很多模型都采用了多查询注意力机制。
	- ![研究结果表明](https://arxiv.org/abs/2305.13245)，可以通过对已经训练好的模型进行微调来添加多查询注意力支持，仅需要约 5% 的原始训练数据量就可以达到不错的效果。
	- ```python
	  # Multi Head Attention
	  # Multi-Head Attention 的创建方法
	  # 查询、键和值 3 个矩阵, 所以是 3 * d_model
	  # 每个 tensor 都是 (1, 512, 768)
	  
	  self.Wqkv = nn.Linear(self.d_model, 3 * self.d_model, device=device)
	  query, key, value = qkv.chunk(3, dim=2)
	  # Multi Query Attention
	  # Multi-Query Attention 的创建方法
	  # 只创建查询的头向量，所以是 1* d_model
	  # 而键和值不再具备单独的头向量
	          # query -> (1, 512, 768)
	          # key   -> (1, 512, 96）
	          # value -> (1, 512, 96)
	  self.Wqkv = nn.Linear(d_model, d_model + 2 * self.head_dim, device=device,)、
	  query, key, value = qkv.split(
	      [self.d_model, self.head_dim, self.head_dim],
	      dim=2
	  )
	  ```