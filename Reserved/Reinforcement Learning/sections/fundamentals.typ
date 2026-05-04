#import "../generic.typ": *

#import "@preview/cetz:0.4.2"
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

= Fundamentals

== Markov Decision Process (MDP)

马尔可夫决策过程（Markov decision process，MDP）可建模为：$(S, A, p, gamma)$，其中 $S$ 为状态空间，$A$ 为动作空间，$p(r, s' mid(|) s, a)$ 为状态转移概率，$gamma in [0, 1)$ 为折扣因子。不同定义是很多的，比如这里状态转移概率 $p(r, s' mid(|) s, a)$ 的写法比较全面，将奖励 $r$ 统一进去了，可以表达转移后所得奖励也有随机性的情况；有时则定义转移概率为 $p(s' mid(|) s, a)$，而奖励函数 $r(s, a, s')$ 是确定的，都看具体需求。

马尔可夫过程的核心假设是马尔可夫性：

$
p(r_(t+1), s_(t+1) mid(|) s_t, a_t, s_(t-1), a_(t-1), ...)
=
p(r_(t+1), s_(t+1) mid(|) s_t, a_t)
$

即状态转移只与当前状态和当前动作有关，与更早的历史信息无关。

== Trajectory, Reward and Return

不断采取动作、在状态间转移并获取*奖励*（reward），走出的一条轨迹实例可以写作：

$
tau = (s_0, a_0, r_1, s_1, a_1, r_2, s_2, dots)
$

具体地，从 $s_0$ 开始，每一步在状态 $s_t$ 时采取动作 $a_t$，导致转移到 $s_(t+1)$ 同时获取该步的奖励 $r_(t+1)$：

$
s_0
stretch(->, size: #200%)^(a_0)_(r_1)
dots
stretch(->, size: #200%)^(a_(t-1))_(r_t)
s_t
stretch(->, size: #200%)^(a_t)_(r_(t+1))
s_(t+1)
stretch(->, size: #200%)^(a_(t+1))_(r_(t+2))
dots
$

从时刻 $t$ 开始往后的总体奖励称为*回报*（return）：

$
G_t &= r_(t+1) + gamma G_(t+1) \
&= r_(t+1) + gamma r_(t+2) + gamma^2 r_(t+3) + dots \
&= sum_(k=0)^infinity gamma^k r_(t+k+1)
$ <equ:return>

其中折扣因子 $gamma$ 用来刻画 “长期奖励贡献递减” 的特性，不考虑太远的奖励，但也不能只看眼前的奖励。

== Policy and Value/Action-Value Functions

策略用 $pi (a mid(|) s)$ 表示，即在状态 $s$ 下采取动作 $a$ 的概率。根据策略采样就能采出一条轨迹 $tau$，从轨迹上的奖励可以计算出这条轨迹（从某时刻出发）的回报。

但因为我们的建模中策略和状态转移都带有随机性，每次采样得到的轨迹、回报都不一定相同，所以我们需要一个对回报平均值的衡量，也就是期望回报：

$
V^pi (s) := E_(tau~p_tau^pi (dot)) [G_t mid(|) s_t = s], quad forall t
$ <equ:v_func>

我们把符号定义完善一点，上式中期望 $E$ 的下标指示期望计算的随机变量是轨迹 $tau$，其采样自随机分布 $p_tau^pi (dot)$，表示在策略 $pi$ 下轨迹 $tau$ 的概率密度函数。同时，注意条件中的 $t$ 只是一个代表索引的符号，自然也可以是 $i, j, k$ 等，而这个 $t$ 是任意的。

这个期望回报称为*值函数*（value function），自变量是状态 $s$，就表示在使用策略 $pi$ 的前提下从 $s$ 出发的平均回报，是#underline[对给定策略下某个状态平均价值的一种衡量]。

#blockquote([
  *关于值函数与索引 $t$ 无关的证明*：

  值函数的定义中直接阐明了它的值和 $t$ 无关，从理解角度来说这很合理，由马尔可夫性，未来状态都和历史无关，具体什么时候到某个状态并不重要，于是值函数理应是与 $t$ 无关的。

  但定义直接这么写就显得比较奇怪，如同 $t$ 是突然冒出来的一样，所以我们从头换一个说法，先假设值函数与 $t$ 有关，定义：

  $
  V_t^pi (s) := E_(tau~p_tau^pi (dot)) [G_t mid(|) s_t = s]
  $ <equ:v_t_func>

  然后我们在适当假设下去证明它们都相等：
  
  $
  V_t^pi (s) = V^pi (s), quad forall t
  $
  
  即可说明值函数与 $t$ 无关，从而名正言顺地书写前述定义。我们先梳理一下需要的*假设*：1、策略和转移概率模型时不变；2、转移概率满足马尔可夫性；3、折扣回报时域窗口无限长且奖励函数有界。然后从 @equ:return 和 @equ:v_t_func 开始：

  $
  V_t^pi (s)

  &=^("def") E_(tau~p_tau^pi (dot)) [G_t mid(|) s_t = s] \

  &=^((1)) E_(tau~p_tau^pi (dot)) [r_(t+1) + gamma G_(t+1) mid(|) s_t = s] \

  &=^((2)) sum_a pi(a mid(|) s) sum_(r, s') p(r, s' mid(|) s, a) \
  &quad E_(tau~p_tau^pi (dot)) [r_(t+1) + gamma G_(t+1) mid(|) s_t = s, a_t = a, r_(t+1) = r, s_(t+1) = s'] \

  &=^((3)) sum_a pi(a mid(|) s) sum_(r, s') p(r, s' mid(|) s, a) \
  &quad (r + gamma E_(tau~p_tau^pi (dot)) [G_(t+1) mid(|) s_t = s, a_t = a, s_(t+1) = s']) \

  &=^((4)) sum_a pi(a mid(|) s) sum_(r, s') p(r, s' mid(|) s, a) (r + gamma E_(tau~p_tau^pi (dot)) [G_(t+1) mid(|) s_(t+1) = s']) \

  &=^((5)) sum_a pi(a mid(|) s) sum_(r, s') p(r, s' mid(|) s, a) (r + gamma V_(t+1)^pi (s'))
  $ <equ:time_varying_v_func_bellman>

  上式中，$(2)$ 是由策略、模型时不变假设进行条件概率分解；$(3)$ 是代入 $r_(t+1)$ 将常数提出去；$(4)$ 是由马尔可夫性，在已知 $s_(t+1) = s'$ 时求 $G_(t+1)$ 的期望与历史信息无关，故移除两个 $t$ 时刻条件；最终得到 $V_t^pi (s)$ 的递推式。写成算子形式，定义：

  $
  (T^pi f)(s) := sum_a pi(a mid(|) s) sum_(r, s') p(r, s' mid(|) s, a) (r + gamma f(s'))
  $

  于是递推式可写成：
  
  $
  V_t^pi = T^pi V_(t+1)^pi
  $

  第二步我们来证明 $T^pi$ 是一个*压缩映射*（contraction mapping）。任取两个函数 $u, v$，对任意状态 $s$ 有：

  $
  (T^pi u)(s) - (T^pi v)(s) = sum_a pi(a mid(|) s) sum_(r, s') p(r, s' mid(|) s, a) gamma (u(s') - v(s'))
  $

  其中 $r$ 消去了。两侧取绝对值有：

  $
  abs((T^pi u)(s) - (T^pi v)(s)) &= abs(sum_a pi(a mid(|) s) sum_(r, s') p(r, s' mid(|) s, a) gamma (u(s') - v(s'))) \
  &<=^((1)) gamma sum_a pi(a mid(|) s) sum_(r, s') p(r, s' mid(|) s, a) abs(u(s') - v(s')) \
  &<=^((2)) gamma sum_a pi(a mid(|) s) sum_(r, s') p(r, s' mid(|) s, a) norm(u - v)_infinity \
  &=^((3)) gamma norm(u - v)_infinity
  $

  式中，$(1)$ 使用的是三角不等式（和之绝对值不超过绝对值之和）；$(2)$ 是用无穷范数（最大值）作为上界；$(3)$ 是由于 $norm(u - v)_infinity$ 是常数，与求和中的随机变量无关，概率分布求和结果得 $1$。
  
  该不等式对 $forall s$ 成立，故由 $(2)$ 中同样的道理，将左侧替换为上确界得：

  $
  norm(T^pi u - T^pi v)_infinity <= gamma norm(u - v)_infinity
  $ <equ:v_func_contraction_mapping>

  由于 $0 <= gamma < 1$，故 $T^pi$ 在 $norm(dot)_infinity$ 下是一个压缩映射，可以说是函数在映射后的距离会不断缩短。由压缩映射的性质，存在唯一的*不动点*（fixed point）$V^pi$ 满足：

  $
  V^pi = T^pi V^pi quad => quad V^pi = (T^pi)^n V^pi, quad forall n in NN
  $ <equ:v_func_fixed_point>

  到这一步，只能说明存在性（存在这样一个时不变的值函数候选 $V^pi (s)$），还不能说明唯一性（值函数为什么一定就要取它）。比如我们可以随意构造一个反例 $V_t^pi (s) = V^pi (s) + c gamma^(-t)$，即不动点加上一个指数项，也能让递推式成立：

  $
  (T^pi V_t^pi)(s)
  &= sum_a pi(a mid(|) s) sum_(r, s') p(r, s' mid(|) s, a) (r + gamma V_(t+1)^pi (s')) \
  &= sum_a pi(a mid(|) s) sum_(r, s') p(r, s' mid(|) s, a) (r + gamma V^pi (s') + c gamma^(-t)) \
  &= sum_a pi(a mid(|) s) sum_(r, s') p(r, s' mid(|) s, a) (r + gamma V^pi (s')) \
  & quad + sum_a pi(a mid(|) s) sum_(r, s') p(r, s' mid(|) s, a) (c gamma^(-t)) \
  &= V^pi (s) + c gamma^(-t) \
  &= V_t^pi (s)
  $

  所以第三步，我们借助最后一个假设（即无限时域折扣回报，以及奖励函数有界）证明值函数的唯一性。奖励函数有界即奖励绝对值存在上界：

  $
  abs(r_(t+1)) <= r_"max"
  $

  那么对于任意 $t$，折扣回报也有界：

  $
  abs(G_t) <= r_"max" sum_(k=0)^infinity gamma^k = r_"max"/(1 - gamma)
  $

  于是作为回报的期望，值函数也一定有界：

  $
  abs(V_t^pi (s)) = abs(E_(tau~p_tau^pi (dot)) [G_t mid(|) s_t = s]) <= norm(V_t^pi)_infinity <= r_"max"/(1 - gamma), quad forall t
  $

  对值函数的这一层限制是将不动点以外的其他时变递推解排除掉的主因。具体地，值函数迭代 $n$ 步时间有：

  $
  V_t^pi = (T^pi)^n V_(t+n)^pi
  $

  与 @equ:v_func_fixed_point 做差有：

  $
  V_t^pi - V^pi = (T^pi)^n V_(t+n)^pi - (T^pi)^n V^pi
  $

  两侧取无穷范数，再由压缩映射的性质 @equ:v_func_contraction_mapping 有：

  $
  norm(V_t^pi - V^pi)_infinity = norm((T^pi)^n V_(t+n)^pi - (T^pi)^n V^pi)_infinity <= gamma^n norm(V_(t+n)^pi - V^pi)_infinity
  $

  由前值函数的有界性，$V_(t+n)^pi$ 和 $V^pi$ 皆有界，故其差也一定有界，于是：

  $
  exists C, quad norm(V_(t+n)^pi - V^pi)_infinity <= C, quad forall n in NN
  $

  代回前式，由于时域可达无限，取 $n -> infinity$ 时有：

  $
  norm(V_t^pi - V^pi)_infinity <= gamma^n C -> 0
  $

  所以对于任意 $t$ 都有 $V_t^pi = V^pi$，最终证明在适当假设下值函数是独立于 $t$ 的。
])

后续方便起见，我们还可以定义*动作值函数*（action-value function）：

$
Q^pi (s, a) = E_(tau~p_tau^pi (dot)) [G_t mid(|) s_t = s, a_t = a]
$ <equ:q_func>

动作值函数比值函数只是多了 $t$ 时刻采取的动作为 $a$ 的条件，所以含义就是#underline[对给定策略下某个状态时采取某个动作的平均价值]。显然，给定某个状态条件下，动作值函数关于动作 $a$ 的期望就应该是值函数（即在动作维度上做了积分）：

$
V^pi (s) = E_(a~pi(dot mid(|) s)) [Q^pi (s, a)] = sum_a pi(a mid(|) s) Q^pi (s, a)
$ <equ:v_q_func_relation>

其中 $pi(dot mid(|) s)$ 是这个策略在给定状态 $s$ 时选取动作的概率密度函数。点号一般就用来表示自变量，这里明确一点可以再定义一个干净的函数写成 $a~p(x) = pi(x mid(|) s)$，有点柯里化的意思，之后有关此类记号就不再赘述。当然，该关系代入 @equ:q_func_bellman 可得反过来用值函数求动作值函数的：

$
Q^pi (s, a) = sum_(r,s') p(r, s' mid(|) s, a) (r + gamma V^pi (s'))
$ <equ:q_v_func_relation>

总的来说，它们的关系有一种 "穿插、间隔" 的感觉。

顺便定义动作值函数和值函数的差为*优势函数*（advantage）：

$
A^pi (s, a) = Q^pi (s, a) - V^pi (s)
$

指示#underline[在给定状态下采取各个动作 $a$ 的价值比整体平均情况要好多少]，在后文 policy gradient 中会用上。

== Bellman Expectation Equation and Bellman Optimality Equation

在前面对不动点 @equ:v_func_fixed_point 的讨论中已经证明了有关值函数的递推式：

$
V^pi (s) = sum_a pi(a mid(|) s) sum_(r, s') p(r, s' mid(|) s, a) (r + gamma V^pi (s'))
$ <equ:v_func_bellman>

或者写回紧凑的条件期望的形式即 *Bellman 期望方程*（Bellman expectation equation）：

$
V^pi (s) = E_(tau~p_tau^pi (dot)) [r_(t+1) + gamma V^pi (s_(t+1)) mid(|) s_t = s]
$

类比值函数的推导，可以得到动作值函数的版本：

$
Q^pi (s, a)
&= E_(tau~p_tau^pi (dot)) [r_(t+1) + gamma Q^pi (s_(t+1), a_(t+1)) mid(|) s_t = s, a_t = a] \
&= sum_(r,s') p(r, s' mid(|) s, a) (r + gamma sum_(a') pi (a' mid(|) s') Q^pi (s', a')) 
$ <equ:q_func_bellman>

可以代回 @equ:v_q_func_relation 验证。

到现在为止，我们定义的值函数和动作值函数都是基于策略 $pi$ 的，通过调整策略可以得到不同的值函数集合。那自然存在一种最优的策略，使得采用这个策略时，每个状态上的值函数（或每组状态-动作对上的动作值函数）都取到可能的最大值，定义这个最大值为*最优值函数*：

$
V^* (s) := sup_pi V^pi (s)
$ <equ:opt_v_func_def>

和*最优动作值函数*：

$
Q^* (s, a) := sup_pi Q^pi (s, a)
$

// 相应地，最优策略其实也就是在每个状态上都取对应最优动作值函数的那个动作。

需要注意的是，我们前面说最优质函数是 "最大值"，但定义中使用的是上确界 $sup$，严格来说它们并不相同，因为上确界不一定能取到，例如 $(0, 1)$ 的上确界是 $1$。定义中如果要使用 $max$，需要能证明这个最优策略的存在性，所以我们采用更具一般性的 $sup$。不过在大部分 RL 问题场景下都可以通过证明 Bellman 最优算子是压缩映射来证明最优策略存在，详见后文。

对于这两个最优函数有 *Bellman 最优方程*（Bellman optimality equation）：

// $
// V^* (s) = max_pi sum_a pi (a mid(|) s) sum_(r,s') p(r, s' mid(|) s, a) (r + gamma V^* (s'))
// $

// $
// Q^* (s, a) = sum_(r,s') p(r, s' mid(|) s, a) (r + gamma max_pi sum_(a') pi (a' mid(|) s') Q^* (s', a'))
// $

$
V^* (s) = max_a sum_(r,s') p(r, s' mid(|) s, a) (r + gamma V^* (s'))
$ <equ:v_func_bellman_optimality>

$
Q^* (s, a) = sum_(r,s') p(r, s' mid(|) s, a) (r + gamma max_(a') Q^* (s', a'))
$ <equ:q_func_bellman_optimality>

可以从所谓*最优性原理*（principle of optimality）的角度（当前问题的最优则子问题也最优，否则就能构造出更优的子问题策略实现整体更优）理解这两个式子的结构，但证明不是简单代入定义就可以的，详见下文。

#blockquote([
  *关于 Bellman 最优方程的证明*：

  如果我们简单将 @equ:v_func_bellman 代入 @equ:opt_v_func_def 只能得到：
  
  $
  V^* (s) &= max_pi sum_a pi (a mid(|) s) sum_(r,s') p(r, s' mid(|) s, a) (r + gamma V^pi (s')) \
  &= max_a sum_(r,s') p(r, s' mid(|) s, a) (r + gamma V^#Cre($pi$) (s'))
  $ <equ:v_func_not_optimal_bellman>

  这一步化简是由于加权平均取最值等效于直接采取最优的动作；最后所得表达式的右侧是 $V^pi$ 而非 $V^*$，不是 Bellman 最优方程。
  
  实际证明的思路是分别证明等式左侧大于或等于以及小于或等于等式右侧。第一部分先证明 $<=$，首先由最优值函数定义，对任意 $s'$ 有：

  $
  V^pi (s') <= V^* (s')
  $

  代入期望方程即有：

  $
  V^pi (s) &= sum_a pi(a mid(|) s) sum_(r, s') p(r, s' mid(|) s, a) (r + gamma V^pi (s')) \
  &<= sum_a pi(a mid(|) s) sum_(r, s') p(r, s' mid(|) s, a) (r + gamma V^* (s')) \
  &<= max_a sum_(r, s') p(r, s' mid(|) s, a) (r + gamma V^* (s'))
  $

  该式对任意策略 $pi$ 都成立，于是对最优策略也成立：

  $
  sup_pi V^pi (s) = V^* (s) #Cbl($<=$) max_a sum_(r, s') p(r, s' mid(|) s, a) (r + gamma V^* (s'))
  $ <equ:v_func_bellman_optimality_leq>

  第二部分证明 $>=$。首先由于最优值函数是值函数的上确界（严格来说 $max$ 和 $sup$ 不同），于是对于任意 $epsilon>0$，我们可以构造一个局部策略 $pi^epsilon$ 使得：

  $
  V^(pi^epsilon) (s') >= V^* (s') - epsilon
  $

  然后我们构造总策略 $tilde(pi)$：在当前状态 $s$ 固定采取行动 $a$，然后从下一个状态开始按照 $pi^epsilon$ 执行。在这一策略下，其值函数至少可达：

  $
  V^tilde(pi) (s) &>= sum_(r,s') p(r, s' mid(|) s, a) (r + gamma (V^* (s') - epsilon)) \
  &= sum_(r,s') p(r, s' mid(|) s, a) (r + gamma V^* (s')) - sum_(r,s') p(r, s' mid(|) s, a) gamma epsilon \
  &= sum_(r,s') p(r, s' mid(|) s, a) (r + gamma V^* (s')) - gamma epsilon
  $

  又由于最优值函数是所有策略值函数的上确界，也包括 $tilde(pi)$，故：

  $
  V^* (s) >= V^tilde(pi) (s) >= sum_(r,s') p(r, s' mid(|) s, a) (r + gamma V^* (s')) - gamma epsilon
  $

  该式对任意 $a$ 都成立，于是也就对其中最大者成立：

  $
  V^* (s) >= max_a sum_(r,s') p(r, s' mid(|) s, a) (r + gamma V^* (s')) - gamma epsilon
  $

  令 $epsilon -> 0$ 得证：

  $
  V^* (s) #Cbl($>=$) max_a sum_(r,s') p(r, s' mid(|) s, a) (r + gamma V^* (s'))
  $ <equ:v_func_bellman_optimality_geq>

  结合 @equ:v_func_bellman_optimality_leq 和 @equ:v_func_bellman_optimality_geq 即证得 @equ:v_func_bellman_optimality。

  接下来考虑最优动作值函数的 Bellman 最优方程。由值函数和最优值函数的关系 @equ:v_q_func_relation 等，可以得到最优值函数和最优动作值函数的关系：

  $
  V^* (s) = max_a Q^* (s, a)
  $ <equ:optimal_v_q_func_relation>

  证明略，思路同前证明两方向不等式成立。然后是下面这个式子成立：

  $
  Q^* (s, a) = sum_(r,s') p(r, s' mid(|) s, a) (r + gamma V^* (s'))
  $

  证明略，可以从 $Q^* (s, a)$ 的含义上理解，是 "在状态 $s$ 第一步执行动作 $a$，之后都按最优策略执行所得的期望回报"。将 @equ:optimal_v_q_func_relation 代入该式即得 @equ:q_func_bellman_optimality。
])

// #blockquote([
//   *关于最优策略的存在性*：

//   前文关于最优值函数定义用 $sup$ 还是 $max$ 的讨论处提及了最优策略存在性的问题，即确保最优策略存在才能使用 $max$，否则不能确保 $V^* (s)$ 可以被取到。

//   定义 Bellman 最优算子：

//   $
//   (T^* f)(s) := max_a sum_(r,s') p(r, s' mid(|) s, a) (r + gamma f (s'))
//   $

//   类比前文 "关于值函数与索引 $t$ 无关的证明" 中的思路，证明 $T^*$ 在适当假设下是压缩映射，从而证明不动点 $V^* (s)$ 的存在性。这时候我们只要采用贪心策略选取：

//   $
//   a^* (s) in arg max_a sum_(r,s') p(r, s' mid(|) s, a) (r + gamma V^* (s'))
//   $

//   就能构造出值函数 $V^(pi^*) (s) = V^* (s)$ 的最优策略 $pi^*$。

//   TODO！这里其实没有证明不动点 $V^*$ 就是最优值函数，没和它的定义牵扯到一起，只是符号一样。
// ])
