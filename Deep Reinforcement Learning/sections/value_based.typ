#import "../generic.typ": *

#import "@preview/cetz:0.4.2"
#import "@preview/cetz-plot:0.1.3"
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

#import "@preview/codly:1.3.0": *

#show: codly-init.with()

= Value-Based Learning

== Policy Iteration and Value Iteration <sec:policy_iter_value_iter>

（略）以迭代的方式求解最优 $V^* (s)$，在其基础上贪心制定最优策略。

// 课程跳过这个，毕竟不 deep。如果要写可写到 value iteration 收敛性的证明，涉及 Bellman 算子压缩映射和不动点。

== Tabular Q-Learning and SARSA <sec:tab_ql_and_sarsa>

既然可以迭代求解 $V^* (s)$，索性直接求 $Q^* (s, a)$，更方便求解策略 $a^* = arg max_a Q^* (s, a)$。考虑状态和动作空间离散的情况，$Q (s, a)$ 实际上是表格式的（tabular），包含共 $abs(cal(S)) times abs(cal(A))$ 个元素。

过程类似值迭代，先初始化：

$
Q (s, a) <- 1, quad forall s, a
$

让智能体在环境中进行探索，用*有限差分*（temporal difference，TD）目标进行更新。具体地，任意 $t$ 时刻发生状态转移时得到一组新数据 $(s_t, a_t, r_t, s_(t+1))$，定义目标：

$
y_t := r_t + gamma max_a Q (s_(t+1), a)
$ <equ:tab_ql_target>

此处的*目标*（target）可视作对真实最优动作值函数 $Q^* (s_t, a_t)$ 的估计（节末注释中会更详细地讨论），用它搭配一个学习率 $alpha$ 对当前 $Q (s_t, a_t)$ 进行更新：

$
Q (s_t, a_t) <- (1 - alpha) Q (s_t, a_t) + alpha y_t
$ <equ:tab_ql_update_interp>

这种形式形似 “插值”，或者另一种更像 “更新” 的等价形式也比较常用：

$
Q (s_t, a_t) <- Q (s_t, a_t) + alpha (y_t - Q (s_t, a_t))
$ <equ:tab_ql_update_residue>

其中 $y_t - Q (s_t, a_t)$ 被称为*有限差分误差*（TD error）。// 如此迭代，最终将会有 $Q (s, a) -> Q^* (s, a)$，证明暂略。

值得注意的是，智能体探索时所实际执行的策略即*行为策略*（behavior policy）是基于 $Q$ 估计值计算的，而更新式中 $max_a Q (s_(t+1), a)$ 说明更新时所用的策略即*目标策略*（target policy）是取当前估计下的最优策略。故在 Q-Learning 中，行为策略和目标策略不一致，这种类型的算法称为 off-policy 算法。关于 on-/off-policy 会在后文 @sec:experience_replay 中更详细地讨论。

与 Q-Learning 相对应，自然也有 on-policy 的 SARSA 算法，其每次状态转移采样得到 $(s_t, a_t, r_t, s_(t+1), a_(t+1))$（这五项的首字母也是算法得名原因），选取的目标为：

$
y_t^"SARSA" := r_t + gamma Q (s_(t+1), a_(t+1))
$ <equ:sarsa_target>

该目标直接计算自行为策略采样的样本，用于目标策略的更新，二者一致，故为 on-policy 的。

表格式 Q-Learning 最大的问题在于*维度诅咒*（curse of dimensionality），状态和动作空间较大时 $Q$ 的元素数量指数级上升，让学习几乎成为不可能，也难以处理状态或动作连续的情况。

#blockquote([
  *关于一个有效目标的设计及 Q-Learning 和 SARSA 的收敛性*：

  我们需要澄清一个重要的原则，关于#underline[所谓的目标（target）应该怎样去选取]，以回答问题：为什么前面 SARSA、Q-Learning 等算法采用现在这样的目标？它们为什么有效？

  #underline[一个重要的判断是]：一个有效目标的条件期望必须等同于我们要学习的对象。注意#underline[这不是一个规则]，而是在迭代更新方法框架下得到的一个方便的推论。

  我们得举点例子说明，例如#underline[在 Q-Learning 中]我们希望学习的是最优动作值函数 $Q^* (s, a)$，定义随机变量 $Y_t^* := r(S_t, A_t) + gamma max_a Q^* (S_(t+1), a)$，由最优贝尔曼方程 @equ:q_bellman_optimal 可知其条件期望满足：

  $
  EE_pi [Y_t^* mid(|) S_t = s, A_t = a] = Q^* (s, a)
  $

  即我们要学习的目标。这里 $EE_pi$ 的 $pi$ 可以不写，因为 $(s, a)$ 已经给定，这个期望和策略无关，这里只是强调一下，如果是 $V$ 就无法省略了。于是由 $Y_t^*$ 我们可以设计目标为其样本：
  
  $
  y_t^* := r(s_t, a_t) + gamma max_a Q^* (s_(t+1), a)
  $
  
  然而，这个目标虽然有效，却无法实际应用，因为我们根本不知道 $Q^* (s_(t+1), a)$ 的真值，$Q^*$ 本来就是我们要求的东西。这实际上已经是另一个问题了，解决方法是用当前参数估计 $Q (s_(t+1), a)$ 来近似它，于是就得到了最终 @equ:tab_ql_target 中的 $y_t$。这么做的依据是通过最优贝尔曼算子可以证明 $Q$ 将随迭代收敛到不动点 $Q^*$，#underline[所以 Q-Learning 目标的合理性其实是通过两重依据来保障的]：

  $
  "objective" Q^* =^! EE ["target" Y_t^*] <- EE ["approximated target" Y_t]
  $

  这里等号上加叹号表示我们要求其成立。
  
  我们#underline[再看 SARSA 的例子]，定义 $Y_t^"SARSA" := r(S_t, A_t) + gamma Q^pi (S_(t+1), A_(t+1))$，由贝尔曼期望公式 @equ:q_bellman_expectation，其条件期望满足：

  $
  EE_pi [Y_t^"SARSA" mid(|) S_t = s, A_t = a] = Q^pi (s, a)
  $

  其目标样本即 @equ:sarsa_target 中的形式。但这样我们发现一个问题，SARSA 好像学习的是 $Q^pi$ 而非最优 $Q^*$，也就是说#underline[如果行为策略 $pi$ 是固定的，那么 SARSA 实际上做的就是策略评估（policy evaluation）的工作]。但是没关系，我们会采用例如 $epsilon.alt$-贪婪等探索策略，只要有概率以更贪心的策略去采样，就#underline[相当于是进行了策略优化（policy improvement）]，二者相结合就得到和 @sec:policy_iter_value_iter 中一样的 “策略评估 + 策略优化” 架构，最终将会收敛到最优策略 $pi^*$ 和 $Q^*$。所以 SARSA 的收敛性和最优性其实也是通过两重依据来保障的：

  $
  "optimal" Q^* <- "objective" Q^pi =^! EE ["target" Y_t^"SARSA"]
  $

  之后我们基于相同结构的目标构造损失函数使用残差梯度法更新时，会发现结果是有偏的。#underline[正儿八经证明上述 Q-Learning 和 SARSA 的收敛性和无偏性]还是要通过最优贝尔曼方程或贝尔曼期望方程的去证明不动点的存在性和唯一性。
])

== Value Approximation

为了表格式算法 cannot scale 的问题，我们可以将 $V$ 或 $Q$ 函数参数化，用相对少的参数去表达函数值：

$
v_theta (s) approx V^pi (s), quad q_theta (s, a) approx Q^pi (s, a)
$

其中 $theta$ 表示参数。参数化的方式有很多，例如 SoB（sum of basis）、神经网络等。

参数化后的值函数自然无法再用表格式算法的方式进行更新，通常我们采用梯度更新的方式对参数进行更新。于是我们现在需要考虑的问题变为：1、用什么*损失函数*（loss function）；2、用什么计算*梯度*（gradients）。

=== Dataset

先规范一些符号问题，训练要有数据集，而数据集可以用多种方式表达。我们最关注的往往是状态转移，每次状态转移都会带来一条关于 “在某状态采取某行动会产生什么后果” 的信息，所以在#underline[部分]算法中我们以 $(s_k, a_k, r_k, s_(k+1), a_(k+1))$ 的元组作为最小单位进行存储，而不在意其采样自何种策略、时间戳是多少。比如普通的 Q-Learning 也不在乎 $a_(k+1)$，可以只存：

$
cal(D) = {(s_k, a_k, r_k, s_(k+1))}_(k=1)^n
$

实际实现中通常还会存下一个状态 $s_(k+1)$ 是否是终端状态方便从目标中剔除后续项，例如直接加一个 $"done"_k in {0, 1}$：

$
cal(D) = {(s_k, a_k, r_k, s_(k+1), "done"_k)}_(k=1)^n
$

但这也不绝对，比如说很多 on-policy 算法需要知道样本对应的采样策略，例如之后的 PPO 算法会用到样本对应的采样概率（的对数值）和值函数估计：

$
cal(D) = {(s_k, a_k, r_k, s_(k+1), "done"_k, log pi_"old" (a_k mid(|) s_k), V^"old" (s_k))}_(k=1)^n
$

而且毕竟是 on-policy 算法，也没什么写成集合 $cal(D)$ 的必要，按顺序 $s_(k+1)$ 一般就是下一条数据的 $s_k$，可以不存它。还有一些基于 Monte Carlo 的算法需要整条轨迹信息，需要保持采样轨迹的完整结构：

$
cal(D) = {tau_k}_(k=1)^n, quad tau_k = (s_0, a_0, r_0, s_1, dots, r_(n_k-1), s_(n_k))
$

举例这么多就是说明这需要根据实际需求选择。现在我们的需求是为下面提出的方法铺垫，故作一些简化假设。假设所有样本都采集自固定的策略 $pi$，并且无所谓顺序，以状态转移为最小单元存储：

$
cal(D)_pi = {(s_k, a_k, r_k, s_(k+1), a_(k+1))}_(k=1)^n
$

为了方便证明，我们用随机变量刻画采样过程：

$
D = (S_t, A_t, R_t, S_(t+1), A_(t+1)) ~ P_(cal(D)_pi) (dot)
$

$P_(cal(D)_pi) (dot)$ 是采样分布，只是为了完善记号，之后我们就用 $EE_(cal(D)_pi) [dot]$ 代表 $EE_(D~P_(cal(D)_pi) (dot)) [dot]$。

=== Residual Value Gradients <sec:residual_gradients>

最常用的损失函数仍然是*均方误差*（mean-squared error，MSE），考虑值函数估计 $v_theta (s)$ 和目标之间在数据集 $cal(D)_pi$ 上的均方误差：

$
cal(L)[theta] := EE_(cal(D)_pi) [(overbrace(underbrace(R_t + gamma v_theta (S_(t+1)), "target" Y_t) - v_theta (S_t), "TD-error"))^2]
$ <equ:loss_vfunc_rvg>

动作值函数的版本如下：

$
cal(L)[theta] := EE_(cal(D)_pi) [(overbrace(underbrace(R_t + gamma q_theta (S_(t+1), A_(t+1)), "target" Y_t) - q_theta (S_t, A_t), "TD-error"))^2]
$ <equ:loss_qfunc_rvg>

可以看出它们的设计是对应 on-policy 策略的，而后者实际上就是 SARSA。直接采用梯度下降法优化损失函数，这一套损失函数设计与梯度更新方式称为*残差梯度*（residual gradients）法，平方项内的部分可视为*残差*（residues）。

残差 $Y_t - q_theta (S_t, A_t)$ 的形式和 @equ:tab_ql_update_residue 中 $y_t - Q(s_t, a_t)$ 也是一致的，就如前文所述，$Y_t$ 是条件期望等于欲学习对象的目标，而#underline[残差本质上就是在衡量当前估计与目标的距离]，优化该损失函数就是在尝试推动 $q_theta$ 收敛到欲学习的对象。

但要注意，目标 $Y_t$ 中用到了 $q_theta (S_(t+1), A_(t+1))$，它也是基于当前参数 $theta$ 的一个#underline[近似值而不是真实的] $Q^* (S_(t+1), A_(t+1))$，这导致它实际无法真的令 $q_theta$ 收敛到 $Q^*$。类似这种 “用估计的参数当目标去优化参数估计” 的行为我们称为*自举*（bootstrapping），也是引发问题的根源。细节上的不同使得这种自举迭代可能让结果收敛（例如值迭代和表格式 Q-Learning），也可能导致发散或产生稳态误差。

#underline[在 Q-Learning 的迭代更新框架下可以证明其最终仍然收敛到最优值，但此处残差梯度法的结果略有不同]，接下来就具体讨论这个问题。为了更清晰地展示参数的变化，我们以表格式 Q-Learning 为例进行说明。它也可以视作一种参数化，只不过参数就是所有表格值 $theta := bold(Q) = {Q_(s,a)} in RR^(abs(cal(S)) times abs(cal(A)))$。写出 Q-Learning 观察到转移 $(s_t, a_t, r_t, s_(t+1))$ 后的一步损失：

$
cal(L)[bold(Q)] := (r_t + gamma Q_(s_(t+1), a^*) - Q_(s_t, a_t))^2, quad a^* := arg max_a Q_(s_(t+1), a)
$

及一步梯度下降更新式：

$
bold(Q) <- bold(Q) - alpha/2 nabla_bold(Q) cal(L)[bold(Q)]
$

由于参数就是表格元素，各元素对 $bold(Q)$ 求导就是对应位置的单位冲激函数：

$
nabla_bold(Q) Q_(s, a) = delta_(s,a), quad forall s in cal(S), forall a in cal(A)
$

其中 $delta_(s,a) = 1$ 当且仅当自变量为 $(s, a)$，课件则采用的类似 $delta (a = dot, s = dot)$ 的写法。于是表格每项的更新为：

$
Q_(s, a) <- Q_(s, a) - alpha (r_t + gamma Q_(s_(t+1), a^*) - Q_(s_t, a_t)) (gamma delta_(s_(t+1), a^*) - delta_(s_t, a_t)), quad forall s in cal(S), forall a in cal(A)
$

由于其中两个 $delta$ 函数的存在，这里的一步更新实际上只会更新两个元素：

$
Q_(s_t, a_t) &<- (1 - alpha) Q_(s_t, a_t) + alpha (r_t + gamma Q_(s_(t+1), a^*)) \
Q_(s_(t+1), a^*) &<- (1 - alpha gamma^2) Q_(s_(t+1), a^*) - alpha gamma (r_t - Q_(s_t, a_t))
$

第一部分 $Q_(s_t, a_t)$ 的更新没有什么问题，$r_t + gamma Q_(s_(t+1), a^*)$ 就是 $y_t$，和之前 @equ:tab_ql_update_interp 中完全一致；但这里还多出了第二部分 $Q_(s_(t+1), a^*)$ 的更新，这就是与前文标准的表格式 Q-Learning 不一致的地方了。

这会导致两个问题，#underline[首先最重要的是多了这一项的残差梯度是*有偏的*（biased），无法再令参数收敛到正确值]，证明见后。我们可以先从现有结论上接受这一点：利用贝尔曼最优算子可以证明 $Q^*$ 是最优贝尔曼方程的唯一不动点解，且标准表格式 Q-Learning 通过迭代更新会收敛到这个解；但如前所述，现在残差梯度化简后比迭代更新的式子多了一条，多半结果也会有偏差。

第二是#underline[违背了*因果性*（causality）]。具体地，更新的本质是树立一个目标对当前参数 $Q_(s_t, a_t)$ 进行调整，但残差梯度同时还去更新了未来状态的参数 $Q_(s_(t+1), a^*)$。这种#underline[参数估计和目标同时在变]的情况自然会导致更新过程不稳定，参数信息传播和学习速度变慢。

#blockquote([
  *关于基于一步 TD 目标的残差梯度法在固定策略数据上优化结果有偏的证明*：

  标题叠了一堆甲，主要是关于几点假设：1、用残差梯度法直接优化；2、以值函数采样目标 $y_t := r_t + gamma v_theta (s_(t+1))$ 为例；3、数据集采样自固定策略 $pi$。

  如果按之前讨论的目标设计要求，该目标的条件期望正是我们希望估计的 $V^pi (s)$，没有问题；但我们从迭代更新框架迁移到损失函数优化问题上这一过程只能说是直觉的，实际上没道理认为目标设计一致就能直接用，仍然需要证明，而结果也确有问题。

  接下来我们具体证明直接对 @equ:loss_vfunc_rvg 应用残差梯度更新所得结果是有偏的。我们不是去证明迭代结果偏离，而是证明损失函数有偏（源于目标中采用的是对值函数的估计而非真实值函数），从而说明优化结果有偏。
  
  $
  "TODO Assignment A1.3"
  $

  动作值函数的目标道理也差不多，不再赘述。
])

=== Value Semi-gradients <sec:semi_gradients>

为了解决前述问题，标准 Q-Learning 常采用的是*半梯度*（semi-gradients）更新的方式。比如上节例子中残差梯度法同时更新 $Q_(s_t, a_t)$ 和 $Q_(s_(t+1), a^*)$，半梯度法则只用 $Q_(s_t, a_t)$ 这一部分梯度更新，从而和表格式 Q-Learning 保持一致并得以套用其收敛性结论。

回到参数化损失 @equ:loss_vfunc_rvg，问题本质是目标 $y_t$ 依赖当前参数估计 $theta$，那么我们可以通过将目标中的 $v_theta$ 替换为另一组迟滞参数驱动的 $v_(theta')$ 来削减这种耦合：

$
cal(L)[theta] := EE_(cal(D)_pi) [(underbrace(R_t + gamma v_(theta') (S_(t+1)), "bootstrapped target" Y_t) - v_theta (S_t))^2]
$

这样在更新时就有 $nabla_theta v_(theta') (S_(t+1)) = 0$，对应舍去的那一部分梯度。实际更新时 $theta'$ 的数值是和 $theta$ 相等的（$theta' = theta$），所以说明白点就#underline[只是反向传播时切断了 $theta'$ 的这一部分梯度]，前向传播时照旧。

以上更新方式称为*半梯度有限差分学习*（semi-gradient TD-learning）。在 PyTorch 具体实现中，给相应项加上 `.detach()` 即可实现只截断某一部分反向传播的操作。该方案#underline[比残差梯度法更快且无偏]，对#underline[线性模型可以证明收敛]，例如前文表格 Q-Learning 的例子。

但#underline[采用神经网络等模型参数化值函数时就很容易发散]，这是由于模型估计误差、自举性等因素复杂综合导致的。引入 $theta'$ 只能弱化耦合而无法消除，因为终究无法得到真实的 $Q^*$ 充当优化目标。

若要更进一步，可以考虑*神经拟合 Q 迭代*（neural-fitted Q-iteration，NFQ）算法，这是第一个成功的深度强化学习算法，就是迭代比较慢。相比普通的半梯度学习，它会在每次阶段性收敛后再将 $theta$ 同步到 $theta'$，一步一个脚印显著降低发散的概率。

=== Online Q-Learning <sec:online_q_learning>

接下来讨论一个实例的具体实现细节，考虑用半梯度 TD 学习实现在线 Q-Learning，神经网络作为参数化模型，“在线” 表示从实时探索中采样所得样本中学习。这是最基础的深度 Q-Learning，为后文 DQN 做铺垫。

首先有关参数化模型 $q_theta (s, a)$，自然的设计是将 $s$ 和 $a$ 作为输入层，输出层输出 $Q$ 值。但对于离散动作 Q-Learning，更新时需要反复计算 $max_a q_theta (s_(t+1), a)$，如果能一次前向传播计算出所有动作 $a$ 对应的 $q_theta (s_(t+1), a)$ 值（输出一个向量）相比依次计算最后取最值自然是更高效的。所以我们#underline[将状态作为输入层，输出层则让每个动作对应一个输出（one output/head per action）]。

引入半梯度法后，模型可以表达为：

$
underbrace(q_theta (s_t, a_t), "value") =^! max_pi EE_pi [sum_(k=0)^infinity gamma^k R_(t+k) mid(|) s_t, a_t] approx underbrace(r(s_t, a_t) + gamma EE [max_(a') q_(theta') (S_(t+1), a')], "target")
$

用 PyTorch 实现节选如下：

#codly()
```python
from torch.optim import RMSprop
from torch.nn.functional import mse_loss
optimizer = RMSprop(q.parameters())
while True:
    # sample episode from environment
    state, action, reward, term, next_state = environment.sample(q)
    # compute left and right side of Bellman eq.
    value = q(state).gather(dim = -1, index = action)
    target = reward + gamma * (~term * q(next_state).max(dim = -1)[0])
    # gradient descent step on supervised regression loss
    optimizer.zero_grad()
    mse_loss(value, target.detach()).backward()
    optimizer.step()
```

其中，可以观察到半梯度法中 `target.detach()` 的使用。此外，代码中的 `term` 指示是否是终止状态（terminal state），#underline[确保在终止状态不再考虑未来的奖励]。

== Stabilization Issues and Techniques

前文讨论过，朴素的半梯度在线 Q-Learning 很容易发散。从机器学习的角度来看，这种不稳定性源于其违背了一系列假设，例如：
+ 回归目标不平稳（non-stationary），即前述的自举问题；
+ 探索策略会改变训练数据分布，采样数据也随之改变；
+ 网络随着训练会遗忘旧样本，即灾难性遗忘（catastrophic forgetting）；
+ 状态转移之间强相关，不满足*独立同分布*（independent and identically distributed，i.i.d.）假设。

接下来我们详细讨论其中较关键的一些问题和它们的解决方案。

=== Experience Replay <sec:experience_replay>

在强化学习中没有传统监督学习的训练集和测试集划分环节，所有数据来自与环境交互的采样。采样和更新采用的策略可能一致可能不一致，即前文提过的 on-policy 和 off-policy 采样。

一般我们将希望评估或者优化的策略记为目标策略 $pi$，将探索中实际采用产生样本的策略记为行为策略 $b$。在我们优化目标策略 $pi$ 时，若更新所用的样本也是策略 $b = pi$ 采集的，那么算法就是 on-policy 的，它#underline[相对稳定且若学习 $V$ 只需要考虑状态输入，维度较低]，但若#underline[策略变化就需要重新采样数据]；若所用的样本是某个其他的行为策略 $b != pi$ 采集的，那么算法就是 off-policy 的，#underline[若学习 $Q$ 它需要更大的输入空间（状态和动作）]，但#underline[可以复用旧的或者其他智能体的经验]。

#blockquote([
  *$V$/$Q$ 的选择与 on-/off-policy 的关系*：
  
  学习 $V$ 还是 $Q$ 这件事和 on-/off-policy 本没有直接的联系，但 on-policy 学 $V$ 和 off-policy 学 $Q$ 是比较自然和直接的。

  实际上，区分 on-/off-policy 的根本依据不是看行为策略和目标策略是否一致，#underline[因为理论上 off-policy 的算法都可以 “退化” 到 on-policy]，把实时数据当历史数据用就可以了；但 on-policy 通常不能当 off-policy 算法用，所以我们#underline[需要看的应该是一个算法是否有能力利用不同于目标策略的行为策略产生的样本来学习优化目标策略]。

  而是否能利用其他策略提供的样本，主要就#underline[看代入这个样本之后目标 $y_t$ 是否还合法、有效]（关于合理的样本设计，回顾 @sec:tab_ql_and_sarsa 末注释中的讨论）。
  
  如果要学习 $V^pi (s)$，应设计如 @equ:loss_vfunc_rvg 中的目标 $y_t^pi := r_t + gamma v_theta (s_(t+1))$。由贝尔曼期望公式，对应的随机变量 $Y_t^pi := R_t + gamma v_theta (S_(t+1))$ 的条件期望满足：

  $
  EE_pi [Y_t^pi mid(|) S_t = s] &= EE_pi [R_t + gamma v_theta (S_(t+1)) mid(|) S_t = s] \
  &= sum_a #Cbl($pi(a mid(|) s)$) sum_(s') P(s' mid(|) s, a) [r(s, a) + gamma V^pi (s')] \
  &= V^pi (s)
  $

  其中，有 $A_t ~ pi(dot mid(|) S_t = s)$ 和 $S_(t+1) ~ P(dot mid(|) S_t = s, A_t)$。如果代入由某个行为策略 $b != pi$ 采集的样本，会导致得到的 $Y_t^b$ 的条件期望中 $A_t ~ b(dot mid(|) s)$，而：

  $
  EE_pi [Y_t^b mid(|) S_t = s] &= sum_a #Cbl($b(a mid(|) s)$) sum_(s') P(s' mid(|) s, a) [r(s, a) + gamma V^pi (s')] \
  &!= V^pi (s)
  $

  故 $y_t^b$ 不再是一个能正确学习 $V^pi (s)$ 的目标，所以这种直接 TD 估计 $V^pi (s)$ 的更新式通常只能是 on-policy 的。

  相同的思路下，如果要学习 $Q^pi (s, a)$，计算目标的条件期望时直接给定了条件 $S_t = s, A_t = a$，这使得 $A_t$ 和 $S_(t+1)$ 的#underline[分布与策略无关]：不管是 $pi$ 还是 $b$，所得条件期望都等于 $Q^pi (s, a)$，可以正确地学习到目标函数。理解一下，意思就是随便什么样本，只要确实在状态 $s$ 采取了动作 $a$ 就可以用于更新 $Q (s, a)$，无关是采集的策略是谁。所以 off-policy 的算法学习 $Q$ 更自然。

  这就是为什么课件中总结称 “Values can only be estimated on-policy, Q-values off-policy”，#underline[严格来说这句话不普遍成立]，但确实可以说 $V$ 的直接 TD 估计通常是 on-policy 的，而 $Q$ 也更容易用于 off-policy 算法。

  至于反例当然存在，例如如果我们还是想学习 $V^pi (s)$，但数据不是 $pi$ 而是 $b$ 采集的，那通过*重要性采样*（importance sampling）校正后就可以做 off-policy 了，此处暂不展开；反过来，前文提过 off-policy 的算法本来就可以退化为 on-policy，包括上述学习 $Q$ 的例子。
])

使用最优策略更新的 Q-Learning 是 off-policy 的，我们可以利用这一特性，为其添加一个*经验重放缓存*（experience replay buffer）。它#underline[存储最近 $n$ 次状态转移中得到的数据样本]，每次训练时会#underline[随机从中采集一个小批次]（mini-batch）的数据，并且#underline[始终包括最近采集的样本]。这样做让部分旧样本仍参与学习，#underline[可减轻遗忘的问题]，随机 mini-batches 也让数据分布#underline[更均匀、更接近 i.i.d.]。

重放缓存的一个变体是*优先经验重放缓存*（prioritized experience replay buffer），它会优先存储误差较大的样本。直觉上，这是为了#underline[针对性反复学习之前发现没学好的样本]，但同时这也有可能打乱训练数据分布（误差较大的样本可能在新旧、类型上分布不均），#underline[降低其对灾难性遗忘问题的抵抗力]。

=== Exploration

训练或者说数据采集阶段，让智能体在环境中探索的最朴素的行动策略是纯*贪婪策略*（greedy policy），例如 Q-Learning 中就是取当前状态 $Q$ 值较大的动作。这样做会产生一些问题，首先贪心策略时间一长就会反复走那几条高分路径，#underline[重放缓存中将被相似的样本塞满]；用这些匮乏的样本训练将会#underline[使涉及不到的那些状态-动作对的 $Q$ 值得不到可靠更新从而被 “遗忘”]，失去泛化能力；总是选取最值的贪婪策略还#underline[容易反复选取偶然产生的高分动作导致不断放大错误]。

为缓解这些问题，引入*$epsilon.alt$-贪婪探索策略*（$epsilon.alt$-greedy exploration policy）：在探索时，以 $epsilon.alt$ 的概率进行随机游走，$1 - epsilon.alt$ 的概率采取贪婪策略。同时这个 $epsilon.alt$ 将#underline[随时间线性衰减]，直到 $n_epsilon.alt$ 步后或者低于某个阈值 $epsilon.alt$ 时停止，即前期较为随机、后期趋于贪心。

相较纯贪婪策略，$epsilon.alt$-贪婪策略为探索引入了一定的随机性，令其#underline[一直保持探索的可能]。当然，这是训练过程中采用的策略，实际部署用于#underline[测试时自然还是采用纯贪婪策略，即执行所学的最优策略]。

=== Target Networks

本节还是关于 @sec:residual_gradients 和 @sec:semi_gradients 中老生常谈的参数估计自举问题。由前，半梯度法引入 $theta'$ 以防止目标被同时更新，这里给用这套迟滞参数的模型（网络）起个正式名字叫*目标网络*（target networks）。

目标网络具体如何缓解自举问题就不重复了，前面没有细讲的一个问题是 $theta'$ 和 $theta$ 同步时机的问题。最直截了当的就是 @sec:online_q_learning 中实例的做法，只是截断了目标网络这一项的反向传播，这意味着每次前向传播采用的 $theta$ 和 $theta'$ 都是同步的，这样做其实还是有 “目标和估计一起动” 的问题存在。

要减轻该现象，最简单是可以让目标网络停住一阵子，给当前估计留一些追赶的时间。例如，每 $n = 10$ 步才更新一下 $theta' <- theta$，在此之前都用旧值计算目标，这称为*硬目标更新*（hard target update）。

软目标更新则是类似于套一个低通滤波器，用：

$
theta' <- (1 - eta) theta' + eta theta
$

更新目标网络参数，$eta$ 可以选比如 $0.1$ 之类，这被称为*软目标更新*（soft target update），总体来说效果更好一些。

== Deep Q-Networks (DQN)

*深度 Q 网络*（DQN）包含一系列在线深度 Q-Learning 算法，总之就是用神经网络作为参数化模型，并且#underline[把上节的重放缓存和目标网络等手段都用上以增强稳定性]的 Q-Learning。一轮（episode）更新的 PyTorch 实现框架如下：

#codly()
```python
# Using mini-batch of transitions from replay buffer
batch = self.replay_buffer.sample()
targets = batch['rewards'] + self.gamma * (~batch['terminals'] \
        * self.target_q(batch['next_states']).max(dim = -1)[0])
values = self.q(batch['states']).gather(dim = -1, index = batch['actions'])
# Backpropagate loss
self.optimizer.zero_grad()
mse_loss(values, targets.detach()).backward()
self.optimizer.step()
# Update target network (hard or soft)
self.target_model_update()
```

接下来就举例说明关于 DQN 的一些性质。

首先是#underline[采样和更新频率对 DQN 表现的影响]。我们不必要像前面讲的基础 Q-Learning 一样每采一个样本就更新一次梯度，而是可以一采就采一轮（包含多步）的数据，然后每轮更新 $n$ 次梯度（$n$ updates/episode）；或者一步一步采样，每 $n$ 步采样更新一次梯度（sampling $n$ steps/update）。这两种 $n$ 放到同一个度量下大致就是反比的关系。

对于前者，$n$ 越大训练越不稳定，即训练曲线的方差看起来越大，同时训练更快，但毕竟容易不稳定所以有限度。对于后者自然是相反，$n$ 越大每次一起更新的样本越多，故训练越稳定，但相应学习速度变慢。

然后是#underline[关于学习阶段]（phase），在线训练过程中样本来自智能体的探索，故训练数据的分布不是固定的，智能体能访问到的状态会随着策略变化而改变，这在许多任务中会#underline[使学习过程自发地划分成多个阶段]。如 Lunar Lander 模拟学习探测器降落的例子中，最初几乎是在随机、混乱地操作，直到某时刻学会了悬停才能稳定进入降落行为所在的状态空间区域。而#underline[进入新阶段的标志通常是损失函数不降反增]，因为智能体在新的状态空间区域将学习到新的行为，此即课件总结的 “learning happens in phases, indicated by increasing losses”。

实践中 DQN 的调优需要经验和技巧。当出现失败（failure）情况，即策略无法生效时，通常可以考虑：
+ 提高梯度更新的学习率 $alpha$；
+ 提高 $gamma$ 以提升未来奖励的贡献；
+ 将网络输入归一化（零均值单位方差）；
+ 将奖励归一化到 $[-1, 1] subset RR$（要谨慎）；
+ 提高探索时长；
+ 增加轮次大小；
+ 修改网络结构等。

当出现不稳定（instability）情况，即一段时间后策略不再进行有效的学习或提升时，可以考虑：
+ 降低学习率 $alpha$；
+ 降低 $gamma$ 减少错误的传播；
+ 减缓目标网络的更新；
+ 增加采样批次（mini-batch）大小；
+ 增加重放缓存大小；
+ 提高随机探索概率 $epsilon.alt$ 的最终下限等。

=== TODO
