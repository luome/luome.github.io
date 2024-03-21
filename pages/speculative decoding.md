- speculative decoding使用小模型进行 decode 一定的 token，再将已decode 的 token 送回大模型（oracle model）进行检查，并将检查正确的tokens 加入已经解码的 tokens 中
	- ```python
	  def generate(prompt: str, tokens_to_generate: int, n_draft: int = 8) -> str:
	      tokens: list[int] = tokenize(prompt)
	      for i in range(tokens_to_generate):
	          # generate `n_draft` draft tokens in the usual autoregressive way
	          draft = tokens[:]
	          for _ in range(n_draft):
	              logits = draft_model.forward(draft)
	              draft.append(argmax(logits[-1]))
	          # run the draft tokens through the oracle model all at once
	          logits = model.forward(draft)
	          checked = logits[len(tokens) - 1 :].argmax(-1)
	          # find the index of the first draft/oracle mismatch—we'll accept every
	          # token before it
	          # (the index might be past the end of the draft, if every draft token
	          # was correct)
	          n_accepted = next(
	              idx + 1
	              for idx, (checked, draft) in enumerate(
	                  # we add None here because the oracle model generates one extra
	                  # token (the prediction for the last draft token)
	                  zip(checked, draft[len(tokens) :] + [None])
	              )
	              if checked != draft
	          )
	          tokens.extend(checked[:n_accepted])
	      return detokenize(tokens)
	  ```
- **优化**：使用概率阈值而不是 decode 数量控制draft 解码多少次
	- ```python
	  
	  def speculative_threshold(
	      prompt: str,
	      max_draft: int = 16,
	      threshold: float = 0.4,
	      threshold_all_correct_boost: float = 0.1,
	  ):
	  
	      tokens = encoder.encode(prompt)
	      # homegrown KV cache setup has an `n_tokens` method that returns the length
	      # of the cached sequence, and a `truncate` method to truncate that sequence
	      # to a specific token
	  
	      model_kv = gpt2.KVCache()
	      draft_kv = gpt2.KVCache()
	      while True:
	          # generate up to `max_draft` draft tokens autoregressively, stopping
	          # early if we fall below `threshold`
	          draft = tokens[:]
	          drafted_probs = []
	          for _ in range(max_draft):
	              logits = draft_model.forward(draft[draft_kv.n_tokens() :], draft_kv)
	              next_id = np.argmax(logits[-1])
	              next_prob = gpt2.softmax(logits[-1])[next_id]
	              if not len(drafted_probs):
	                  drafted_probs.append(next_prob)
	              else:
	                  drafted_probs.append(next_prob * drafted_probs[-1])
	              draft.append(int(next_id))
	              if drafted_probs[-1] < threshold:
	                  break
	          n_draft = len(draft) - len(tokens)
	          # run draft tokens through the oracle model
	          logits = model.forward(draft[model_kv.n_tokens() :], model_kv)
	          checked = logits[-n_draft - 1 :].argmax(-1)
	          n_accepted = next(
	              idx + 1
	              for idx, (checked, draft) in enumerate(
	                  zip(checked, draft[len(tokens) :] + [None])
	              )
	              if checked != draft
	          )
	          yield from checked[:n_accepted]
	          tokens.extend(checked[:n_accepted])
	          if n_accepted <= n_draft:
	              # adjust threshold towards prob of last accepted token, if we
	              # ignored any draft tokens
	              threshold = (threshold + drafted_probs[n_accepted - 1]) / 2
	          else:
	              # otherwise, lower the threshold slightly, we're probably being
	              # too conservative
	              threshold -= threshold_all_correct_boost
	          # clamp to avoid pathological thresholds
	          threshold = min(max(threshold, 0.05), 0.95)
	          # don't include oracle token in kv cache
	          model_kv.truncate(len(tokens) - 1)
	          draft_kv.truncate(len(tokens) - 1)
	  ```