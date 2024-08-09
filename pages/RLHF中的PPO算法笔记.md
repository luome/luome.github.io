# 语言模型中的强化学习
环境：在这个情境下，强化学习的“环境”是人类的互动，LLM 根据与人类的对话（即对话历史）来决定下一步该生成什么内容。
策略：接受提示并返回文本序列的语言模型；
行动：在每个时间步，LLM 会根据对话历史和当前状态生成下一个token
奖励机制：环境会根据从人类偏好数据训练的奖励函数为每个动作给予一个奖励
目标：强化学习的目标是找到一种策略，使得 LLM 在整个对话中能够获得最大的累积奖励
![rlhf.png](../assets/rlhf_ppo1.png){:height 295, :width 400}
- # 策略梯度
  策略梯度算法是一种直接优化 agent 策略的强化学习算法，它直接优化从状态到行动的策略，而不是像值函数方法一样学习一个将状态映射到预期累积奖励的函数。策略梯度算法的核心思想是利用梯度上升算法来最优化策略，通过调整策略的参数来最大化回报的期望。一般来说，策略 $\pi$ 会被参数化，用  $\pi(a|s, \theta)$ 来表示，表示的是在 s 状态下采取行动 a 的概率。那么梯度上升方法更新的方式为：
  $$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$$其中 $\alpha$ 为学习率，而 $J(\theta)$表示在使用策略 $\pi_\theta$ 期望的回报。$\nabla_\theta J(\theta)$ 被称为策略梯度。
  策略梯度的通用形式为：$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \Phi_t \right]$$其中 $\Phi_t$ 可以是 $\Phi_t=R(\tau)$ 或者 $\Phi_t = \sum_{t'=t}^T R(s_{t'}, a_{t'})$ 或者 $\Phi_t = \sum_{t'=t}^T R(s_{t'}, a_{t'}) - b(s_t)$ ，这些选择都会得到相同的期望值，尽管有不同的方差。
  通常来说，回报是使用蒙特卡洛采样来计算的，如果回报是好的，那么所有的行为都会通过增加他们的概率而被强化。这种方式是无偏差的，因为我们依靠的是真实的获得的回报，而不是去估计它。但是，这样做具有很高的方差。方差的来源是不同的行动的轨迹可能导致非常多样的回报，因为环境的随机性和策略本身。
  为了降低这种方差，一种通用的方法是在策略梯度更新的过程中，使用优势函数估计，而不直接使用原始的回报。优势函数$A(s_t, a_t)$ 代表在状态  $s_t$ 下特定的行动 $a_t$ 时，与在相同策略条件下的平均行动相比，有多好。
  $$\Phi_t=A(s_t, a_t)$$
  可以通过 $A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$ 计算，其中 $Q(s_t, a_t)$ 是行动-值函数，表示的是在 $s_t$ 状态下采取行为 $a_t$ 后的平均回报，而 $V(s_t)$ 表示的是值函数，即是在 $s_t$ 状态下的平均期望回报。
  优势函数的估计方法在不同的算法中是不同的，而 Generalized Advantage Estimation (GAE) 是一种常用的算法。
- # 广义优势估计 GAE 
  优势函数 A 是通过 Q 函数与价值函数之间的差值得到的。Q 函数考虑的是一个特定的动作，而价值函数则是根据策略对所有可能的行动进行平均。然而，在实际应用中，我们使用实际情节中的回报（奖励的总和）来估计 Q 函数。由于未来的回报可能会非常不稳定，这会引入大量的方差。减少这种噪音的一种方法是使用价值函数来估计未来的回报（从时间步 t 开始之后）。GAE 算法实际上是在简单的一步时序差分（TD）回报和完整的蒙特卡罗回报之间平衡偏差和方差。以下是对 GAE 推导过程的通俗解释。
  $TD-k$ 回报 $\hat{R}^k_t$ 是实际奖励和估计回报的组合：
  $$\hat{R}^k_t = r_t + \gamma r_{t+1} + ... + \gamma^{(k-1)}r_{t+k-1} + \gamma^k V(s_{t+k})$$
  其中 $\gamma$ 是折扣因子。使用 TD-k 回报的优势估计被称为 k 步优势，其定义为：
  $$A^k_t = \hat{R}^k_t - V(s_t) = \sum^k_{i=1} \gamma^i \delta_{t+i} = -V(s_t) + r_t + \gamma r_{t+1} + ... + \gamma^{(k-1)}r_{t+k-1} + \gamma^k V(s_{t+k})$$
  其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是差分误差。使用 k 步优势存在显著的偏差-方差权衡。如果 k 较小，偏差会较高，因为优势估计是基于更少的步骤，因此严重依赖于价值函数的准确性。另一方面，如果 k 较大，方差可能较高，因为优势估计涉及汇总许多噪声奖励。所以在较为稳定的环境中，可以选择较大的 k 值，以利用更多的回报信息，减少偏差。在较为不稳定或高噪声的环境中，可以选择较小的 k 值，以减少方差，增强稳定性。
  为了平衡优势估计中的偏差-方差权衡，GAE 将优势函数定义为 k 步优势的指数移动平均，权重为$(1-\lambda)\lambda^{(k-1)}$:
  $$\begin{aligned} 
  A^{GAE}(\gamma,\lambda)_t &= (1 - \lambda)(A^{(1)}_t + \lambda A^{(2)}_t + \lambda^2 A^{(3)}_t + \cdots) \\ 
  &= (1 - \lambda)(\delta_t + \lambda(\delta_t + \gamma\delta_{t+1}) + \lambda^2(\delta_t + \gamma\delta_{t+1} + \gamma^2\delta_{t+2}) + \cdots)\\
  &= (1 - \lambda)(\delta_t(1 + \lambda + \lambda^2 + ...) + \gamma\delta_{t+1}(\lambda + \lambda^2 + \lambda^3 + \cdots) + \gamma^2\delta_{t+2}(\lambda^2 + \lambda^3 + \lambda^4 + \cdots) + \cdots)\\
  &= (1 - \lambda)\left(\frac{\delta_t}{1-\lambda} + \frac{\gamma\delta_{t+1}}{1-\lambda} + \frac{\gamma^2\delta_{t+2}}{1-\lambda} + \cdots \right)\\
  &= \sum_{i=0}^{\infty}(\gamma\lambda)^i\delta_{t+i}
  \end{aligned}
  $$
  GAE 的定义平滑地在高偏差（$\lambda = 0$）与高方差($\lambda=1$)差值，很有效地管理了偏差和方差的 trade-off。
  
  $$GAE(\gamma,0) : \hat{A}_t = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$
  $$GAE(\gamma,1) : \hat{A}_t = \sum_{i=0}^{\infty}\gamma^i\delta_{t+i} = \sum_{i=0}^{\infty}\gamma^i r_{t+i} - V(s_t)$$
  通过GAE，我们可以准确地估计优势函数 $A(s_t, a_t)$ 的 $\hat{A}_t$。这个估计在构建策略梯度估计器时将起到至关重要的作用:
  $$
  \begin{aligned}
  \nabla_{\theta}J(\theta) &= \frac{1}{|D|}\sum_{\tau\in D}\sum_{t=1}^{T}\nabla_{\theta}\log\pi_{\theta}(a_t|s_t)\hat{A}_t \\
  & = \hat{\mathbb{E}}_{t} \nabla_{\theta}\log\pi_{\theta}(a_t|s_t)\hat{A}_t
  \end{aligned}$$
  其中 D 是一个有限的样本集。
  在`trl`包中用简单的几行代码实现了GAE，通过循环和递归方式计算，最终得到每个时间步的优势估计：
  
  ```python
  def compute_advantages(  
    self,  
    values: torch.FloatTensor,  
    rewards: torch.FloatTensor,  
    mask: torch.FloatTensor,  
  ):   
    lastgaelam = 0  # 初始化最后一步的广义优势估计量（GAE）
    advantages_reversed = []  # 用于存储反转后的优势估计
    gen_len = rewards.shape[-1]  # 序列长度
    
    # 应用掩码处理值和奖励
    values = values * mask  
    rewards = rewards * mask  
    if self.config.whiten_rewards:  
        rewards = masked_whiten(rewards, mask, shift_mean=False)  
    
    # 反向遍历每个时间步计算优势
    for t in reversed(range(gen_len)):  
        # 计算下一个时间步的值
        nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0  
        # 计算时序差分误差 
        delta = rewards[:, t] + self.config.gamma * nextvalues - values[:, t]  
        # 计算当前时间步的广义优势估计量
        lastgaelam = delta + self.config.gamma * self.config.lam * lastgaelam  
        # 将计算的优势添加到列表中
        advantages_reversed.append(lastgaelam)  
    
    # 将反转的优势列表恢复正常顺序，并转置为原来的形状
    advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)  
    
    # 计算回报
    returns = advantages + values  
    advantages = masked_whiten(advantages, mask)  
    advantages = advantages.detach()  
    return values, advantages, returns
  ```
- # 近端策略优化
  PPO 和 TRPO 是强化学习（RL）中的两项关键技术，它们能够有效地训练策略，并保持其稳定性。这两个方法的基本理念是“稳步、小步前进”：即通过轻推策略逐步优化，而不是采用激进的更新方式，这种方式可能会破坏整个学习过程的稳定性。
  在传统的强化学习中，策略梯度的原则要求新旧策略在参数空间内保持接近。然而，这种在参数空间内的接近并不一定意味着性能上的相似，参数的微小变化可能会显著影响策略的有效性。此外，如果进行一个大幅度且不受限制的步骤，可能会导致策略性能的崩溃，这种情况通常被称为“falling off the cliff”。这一固有风险限制了原始策略梯度的样本效率。
  TRPO 没有通过参数空间接近，而是在策略更新中通过引入另一种不同的限制，它通过把策略的变化通过 KL 散度进行限制，保留了一个可以接受的限制：
  $$\text{maximize}_{\theta} \, \mathbb{E}_t \left[ \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \hat{A}_t \right] \ 
  \text{subject to}\ \mathbb{E}_t \left[ \text{KL}(\pi_{\theta_{\text{old}}}(\cdot|s_t), \pi_{\theta}(\cdot|s_t)) \right] \leq \delta$$
  
  其中 $\theta_{old}$ 是更新之前的老的策略参数。
  TRPO 将 KL 散度作为硬性限制来阻止策略的有害的更新，而 PPO 有两种基础的变形来防止有害的策略更新：PPO-Penalty 和 PPO-Clip，其中 PPO-Penalty 通过采用基于惩罚的方法而不是约束来解决无约束优化问题：
  $$\mathcal{L}_{\text{ppo-penalty}}(\theta) = \mathbb{E}_t \left[ \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \hat{A}_t \right] - \beta \text{KL}(\pi_{\theta_{\text{old}}}(\cdot|s_t), \pi_{\theta}(\cdot|s_t))$$
  其中 $\beta$ 是惩罚因子。
  
  **PPO-Clip**: PPO-Clip 在目标函数中使用了策略比率的裁剪版本，其目标函数可以被表示为:
  $$   \mathcal{L}_{\text{ppo-clip}}(\theta) = \mathbb{E}_t \left[ \min \left( \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \hat{A}_t, \, \text{clip} \left( \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}, 1 - \epsilon, 1 + \epsilon \right) \hat{A}_t \right) \right]$$
  其中 $\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ 是新策略与旧策略的概率之比，clip 函数把这个值限制在一个区间内，$\epsilon$ 是一个超参数，用来决定新策略与旧策略之间可以偏离的程度。裁剪起到正则化的作用，限制策略在每次迭代中发生剧烈变化的程度。防止过大的策略更新可以确保学习过程的稳健性，同时比普通策略梯度方法保持更高的样本效率。
  
  **值函数估计**：在 PPO 算法中，在 PPO 算法中，评论家模型（通常称为值函数）估计每个状态的预期回报。该模型的学习目标是最小化其预测值与实际回报值之间的差异。评论家模型的损失函数通常使用均方误差（MSE）定义，具体公式如下：
  $$   \mathcal{L}_{\text{critic}}(\phi) = \mathbb{E}_t \left[ \left( V_{\phi}(s_t) - \hat{R}_t \right)^2 \right]$$
  其中 $V_\phi({S_{t}})$ 表示的是评论家模型用参数 $\phi$  在状态 $s_{t}$ 下的预测的值，而 $\hat{R}_t$ 则表示在状态 $s_{t}$ 下真正的回报的值。
  
  在 `trl` 中，这部分两部分的 Loss 代码如下：
  
  ```python
  def loss(  
    self,  
    old_logprobs: torch.FloatTensor,  
    values: torch.FloatTensor,  
    logits: torch.FloatTensor,  
    vpreds: torch.FloatTensor,  
    logprobs: torch.FloatTensor,  
    mask: torch.LongTensor,  
    advantages: torch.FloatTensor,  
    returns: torch.FloatTensor,  
  ):   
    
    vpredclipped = clip_by_value(  
        vpreds,  
        values - self.config.cliprange_value,  
        values + self.config.cliprange_value,  
    )  
    # 计算评论家模型的Loss，对应公式L_critic
    vf_losses1 = (vpreds - returns) ** 2  
    vf_losses2 = (vpredclipped - returns) ** 2  
    vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)  
    vf_clipfrac = masked_mean(torch.gt(vf_losses2, vf_losses1).float(), mask)  
    
    # 计算 ppo-clip
    ratio = torch.exp(logprobs - old_logprobs)  
    pg_losses = -advantages * ratio  
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.config.cliprange, 1.0 + self.config.cliprange)  
    pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mask)  
    
    pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).float(), mask)  
  
    loss = pg_loss + self.config.vf_coef * vf_loss  
  
    avg_ratio = masked_mean(ratio, mask).item() 
  
    # KL 不能过大
    if avg_ratio > self.config.ratio_threshold:  
        warnings.warn(  
            f"The average ratio of batch ({avg_ratio:.2f}) exceeds threshold {self.config.ratio_threshold:.2f}. Skipping batch."  
        )  
        pg_loss = pg_loss * 0.0  
        vf_loss = vf_loss * 0.0  
        loss = loss * 0.0  
  
    entropy = masked_mean(entropy_from_logits(logits), mask)  
  
    approxkl = 0.5 * masked_mean((logprobs - old_logprobs) ** 2, mask)  
    # policykl的作用：早停：如果策略的 KL 大于目标 KL，则将梯度置零，并跳过优化步骤。
    policykl = masked_mean(old_logprobs - logprobs, mask)  
  
    return_mean, return_var = masked_mean(returns, mask), masked_var(returns, mask)  
    value_mean, value_var = masked_mean(values, mask), masked_var(values, mask)  
  
    stats = dict(  
        loss=dict(policy=pg_loss.detach(), value=vf_loss.detach(), total=loss.detach()),  
        policy=dict(  
            entropy=entropy.detach(),  
            approxkl=approxkl.detach(),  
            policykl=policykl.detach(),  
            clipfrac=pg_clipfrac.detach(),  
            advantages=advantages.detach(),  
            advantages_mean=masked_mean(advantages, mask).detach(),  
            ratio=ratio.detach(),  
        ),  
        returns=dict(mean=return_mean.detach(), var=return_var.detach()),  
        val=dict(  
            vpred=masked_mean(vpreds, mask).detach(),  
            error=masked_mean((vpreds - returns) ** 2, mask).detach(),  
            clipfrac=vf_clipfrac.detach(),  
            mean=value_mean.detach(),  
            var=value_var.detach(),  
        ),  
    )  
    return pg_loss, self.config.vf_coef * vf_loss, flatten_dict(stats)
  ```
- # 值函数建模
  值函数 $V_\phi({S_{t}})$ 的建模的做法是在 LLM 上使用一个 `ValueHead` 网络来预测当前步的期望回报。如下面的代码所示：
  
  ```python
  class ValueHead(nn.Module):  
    r"""  
    The ValueHead class implements a head for GPT2 that returns a scalar for each output token.    """  
    def __init__(self, config, **kwargs):  
        super().__init__()  
        if not hasattr(config, "summary_dropout_prob"):  
            summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)  
        else:  
            summary_dropout_prob = config.summary_dropout_prob  
  
        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()  
  
        # some models such as OPT have a projection layer before the word embeddings - e.g. OPT-350m  
        if hasattr(config, "hidden_size"):  
            hidden_size = config.hidden_size  
        if hasattr(config, "word_embed_proj_dim"):  
            hidden_size = config.word_embed_proj_dim  
        elif hasattr(config, "is_encoder_decoder"):  
            if config.is_encoder_decoder and hasattr(config, "decoder"):  
                if hasattr(config.decoder, "hidden_size"):  
                    hidden_size = config.decoder.hidden_size  
  
        self.summary = nn.Linear(hidden_size, 1)  
  
        self.flatten = nn.Flatten()  
  
    def forward(self, hidden_states):  
        output = self.dropout(hidden_states)  
  
        # For now force upcast in fp32 if needed. Let's keep the  
        # output in fp32 for numerical stability.        if output.dtype != self.summary.weight.dtype:  
            output = output.to(self.summary.weight.dtype)  
  
        output = self.summary(output)  
        return output
  ```
- # 算法流程
  ---
  **输入：** 初始策略参数 $\theta_0$，初始价值函数参数 $\phi_0$。
  **for** \(n = 0, 1, 2, \dots\) **do**
  收集一组轨迹 $D_n = \{\tau_i\}$，通过在环境中执行策略 $\pi(\theta_n)$。
  计算回报 $\hat{R}_t$。
  基于当前价值函数 $V_{\phi_n}$，使用广义优势估计方法计算优势估计$\hat{A}_t$。
  通过最大化 PPO-penalty/clip 目标函数来更新策略：$$\theta_{n+1} = \arg \max_{\theta} \mathcal{L}_{\text{ppo-clip}}(\theta_n)$$
    通过最小化均方误差更新价值函数：$$\phi_{n+1} = \arg \min_{\phi}\mathcal{L}_{\text{critic}}(\phi_n)$$
  **end for**
  ---
  
  上面的算法在`trl`包中的实现如下，省略了多余的代码，并加了一些必要的注释：
  
  ```python
  # 计算 LLM 得到当前策略下的语言模型的概率、预测的回报值 
  with torch.no_grad():  
    all_logprobs, logits_or_none, values, masks = self.batched_forward_pass(  
        self.model,  
        queries,  
        responses,  
        model_inputs,  
        response_masks=response_masks,  
        return_logits=full_kl_penalty,  
    )  
    with self.optional_peft_ctx():  
        ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(  
            self.model if self.is_peft_model else self.ref_model,  
            queries,  
            responses,  
            model_inputs,  
            return_logits=full_kl_penalty,  
        )  
  
  timing["time/ppo/forward_pass"] = time.time() - t  
  
  # 计算奖励，从reward model 中进行计算
  with torch.no_grad():  
    t = time.time()  
    if full_kl_penalty:  
        active_full_logprobs = logprobs_from_logits(logits_or_none, None, gather=False)  
        ref_full_logprobs = logprobs_from_logits(ref_logits_or_none, None, gather=False)  
        rewards, non_score_reward, kls = self.compute_rewards(  
            scores, active_full_logprobs, ref_full_logprobs, masks  
        )  
    else:  
        rewards, non_score_reward, kls = self.compute_rewards(scores, all_logprobs, ref_logprobs, masks)  
    timing["time/ppo/compute_rewards"] = time.time() - t  
    # 计算优势与回报
    t = time.time()  
    values, advantages, returns = self.compute_advantages(values, rewards, masks)  
    timing["time/ppo/compute_advantages"] = time.time() - t  
  
  # upcast to float32 to avoid dataset issues  
  batch_dict = {  
    "queries": queries,  
    "responses": responses,  
    "logprobs": all_logprobs.to(torch.float32),  
    "values": values.to(torch.float32),  
    "masks": masks,  
    "advantages": advantages,  
    "returns": returns,  
  }  
  batch_dict.update(model_inputs)  
  
  t = time.time()  
  all_stats = []  
  early_stop = False  
  
  # 进入 PPO 的训练循环
  for _ in range(self.config.ppo_epochs):  
    if early_stop:  
        break  
    b_inds = np.random.permutation(bs)  
    for backward_batch_start in range(0, bs, self.config.backward_batch_size):  
        backward_batch_end = backward_batch_start + self.config.backward_batch_size  
        backward_batch_inds = b_inds[backward_batch_start:backward_batch_end]  
  
        for mini_batch_start in range(0, self.config.backward_batch_size, self.config.mini_batch_size):  
            mini_batch_end = mini_batch_start + self.config.mini_batch_size  
            mini_batch_inds = backward_batch_inds[mini_batch_start:mini_batch_end]  
            mini_batch_dict = {  
                "logprobs": batch_dict["logprobs"][mini_batch_inds],  
                "values": batch_dict["values"][mini_batch_inds],  
                "masks": batch_dict["masks"][mini_batch_inds],  
                # hacks: the queries and responses are ragged.  
                "queries": [batch_dict["queries"][i] for i in mini_batch_inds],  
                "responses": [batch_dict["responses"][i] for i in mini_batch_inds],  
                "advantages": batch_dict["advantages"][mini_batch_inds],  
                "returns": batch_dict["returns"][mini_batch_inds],  
            }  
            for k in model_inputs_names:  
                mini_batch_dict[k] = batch_dict[k][mini_batch_inds]  
            with self.accelerator.accumulate(self.model):  
                model_inputs = {k: mini_batch_dict[k] for k in model_inputs_names}  
                logprobs, logits, vpreds, _ = self.batched_forward_pass(  
                    self.model,  
                    mini_batch_dict["queries"],  
                    mini_batch_dict["responses"],  
                    model_inputs,  
                    return_logits=True,  
                )  
                # 计算 Loss 并执行梯度下降
                train_stats = self.train_minibatch(  
                    mini_batch_dict["logprobs"],  
                    mini_batch_dict["values"],  
                    logprobs,  
                    logits,  
                    vpreds,  
                    mini_batch_dict["masks"],  
                    mini_batch_dict["advantages"],  
                    mini_batch_dict["returns"],  
                )  
                all_stats.append(train_stats)  
  
    # typically, early stopping is done at the epoch level  
    if self.config.early_stopping:  
        policykl = train_stats["policy/policykl"]  
        early_stop = self._early_stop(policykl)  
        if early_stop:  
            break
  ```
-