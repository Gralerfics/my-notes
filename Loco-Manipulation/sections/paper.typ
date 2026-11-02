#import "../generic.typ": *

#import "@preview/cetz:0.4.2"
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

= Papers

== [++] Deep WBC (2022)

=== Introduction

较早的重要工作，通过强化学习得到一个操作和移动的统一策略。Introduction 中先讨论臂和腿间存在的协同关系，认为二者协同控制是更合理的，可以更好发挥二者各自功能，例如通过基座的俯仰扩大操作臂的工作空间；顺势提出全身控制下 sim2real 较传统 locomotion 存在更多挑战。

=== Method

文章综合来说是要学习一个策略同时控制腿和臂以执行身体速度和机械臂末端位姿指令。算上目标指令，*策略的输入*包括：
1. 当前（$t$ 时刻）身体状态 $s_t^"base" in RR^5$（roll、pitch 和三个角速度）；
2. 臂状态 $s_t^"arm" in RR^12$（六自由度关节角和关节速度）；
3. 腿状态 $s_t^"leg" in RR^28$（四个 foot contact indicators 加上十二个腿关节的关节角和关节速度）；
4. 上一时刻动作 $a_(t-1) in RR^18$（六个臂关节加上十二个腿关节关节角）；
5. 机械臂末端位姿指令 $[p^"cmd", o^"cmd"] in SS EE (3)$；
6. 身体速度指令 $[v_x^"cmd", omega_"yaw"^"cmd"]$；
7. 环境外参（extrinsics）$z_t in RR^20$。

*输出*就是动作 $[a_t^"arm", a_t^"leg"] in RR^(6+12)$，通过 PD 控制器（关节位置控制）转为力矩指令。

策略使用*强化学习*方法训练，最大化期望折扣回报 $EE_pi [sum_(t=0)^(T-1) gamma^t r_t]$，其中*奖励*包括操作 $r^"manip"$ 和移动 $r^"loco"$ 两部分，各部分分别有 $r_"following" + r_"energy" + r_"alive"$，具体构成详见文章 Table 1。

训练过程中通过采样、插值取得目标指令作为引导，具体方式略。

==== Advantage Mixing

臂和腿虽然存在协同作用，但*大体上来说还是臂负责操作和腿负责移动的成分居多*，而直接将动作空间拼接统一训练带来的高自由度让我们平白承担了搜索空间的指数扩大。

课程学习是常用的训练加速技巧，但经常需要精心设计和调整。文章提出只设计一种课程，用单个课程参数 $beta$ 让臂和腿先各学各的，再逐渐学习更难的协同操作，称其为*优势混合*（advantage mixing）：

$
&J (theta_pi) \
=&1/abs(cal(D)) sum_((s_t, a_t) in cal(D)) log pi (a_t^"arm" mid(|) s_t) (A^"manip" + beta A^"loco") + log pi (a_t^"leg" mid(|) s_t) (beta A^"manip" + A^"loco")
$

其中 $beta in [0, 1]$，课程中令其随时间线性增长，即 $beta = min (t\/T_"mix", 1)$；$A^"manip"$ 和 $A^"loco"$ 分别是来自奖励 $r^"manip"$ 和 $r^"loco"$ 的优势函数。文章采用 PPO 优化此 RL 目标。

==== Regularized Online Adaptation

另一个问题是 sim2real gap，很多工作采用两阶段的*教师-学生*方案来训练，先用特权信息（privileged information）训练教师策略，再基于可获取的观测历史模仿教师策略训练学生策略。单纯这样做可能导致可实现性（realizability）的 gap，教师可能训练出一种很好用但学生无法从历史中推断出来、无法模仿的表示。

文章提出 ROA，具体来说是训练一个 privileged info encoder $mu$，将特权信息编码为环境外参 $z^mu$，把它塞给统一策略进行端到端的训练（前面提到过这是策略输入的一部分）；同时训练另一个 adaptation module $phi.alt$，将可获取的历史观测编码为 $z^(phi.alt)$，学习 $z^mu$，同时正则化让 $z_mu$ 不能离自己太远来抑制 realizability gap。最终整体的 loss 可以写为：

$
L (theta_pi, theta_mu, theta_(phi.alt)) = -J (theta_pi, theta_mu) + lambda norm(z^mu - "sg"[z^(phi.alt)])_2 + norm("sg"[z^mu] - z^(phi.alt))_2
$

第一项来自端到端 RL 奖励（见前，加入了提供环境外参所需的 $theta_mu$），然后是 regularize 和 supervise 项，$"sg"[dot]$ 是 stop gradient operator，$lambda$ 是作为正则化强度的 Lagrangian multiplier。

文章采取类似对偶优化的思路，称 dual gradient descent，例如将 $norm(z^mu - "sg"[z^(phi.alt)])_2 <= 0$ 视为约束，降低损失，并在约束被违反时升高其权重 $lambda$，交替进行：

$
theta_pi, theta_mu &<- arg min_(theta_pi, theta_mu) EE_((s, a) ~ pi (dots, z^mu)) [L] \
theta_(phi.alt) &<- arg min_(theta_(phi.alt)) EE_((s, a) ~ pi (dots, z^(phi.alt))) [L] \
lambda &<- lambda + alpha (partial L)/(partial lambda)
$

但文章实际上没有完全按照这个步骤训练，而是每特定训练步数后切换，并且令 $lambda$ 在过程中线性从 $0$ 提高到 $1$。文章附录 B 有详细伪代码，是每降低 $H = 20$ 步 $pi$ 和 $mu$ 的损失后正则化一次 $phi.alt$，并且每步略微提高 $lambda$。

=== Experiment

==== Simulation

提出指标：生存率、身体加速度、速度目标误差、末端目标误差、总能量，除了第一个都是越低越好。

实验比较：
1. 文章方案；
2. 腿和臂分别使用单独的策略；
3. 文章方案但腿和臂各用各的奖励，不训练协同；
4. RMA，两阶段教师-学生方案基线；
5. 采用特权信息训练的专家策略；
6. 域随机化策略。

*域随机化*策略在身体加速度和总能量上表现最优，其他都是文章方案最优。

==== Real-World

实物实验对比文章方案和身体 MPC + 臂 IK 方案。实机部署时 commands 来自遥操、视觉或演示等接口，环境外参来自训练好的 adaptation module。后面一些实验主要就是展示协同功能。

=== Remarks

1. 关注两个重要的 method（AM 和 ROA）；
2. RL 的奖励设计；
3. ROA 中的优化背景知识和化用方法；
4. 文章 ROA 处还提到 RMA 是 ROA 的一个特例，这种讲述方式或许好过比如从 RMA 引出新方法；
5. 文章实验部分的写法：先提出问题，问题主要关注重点 method 是否有提升表现、解决其预期解决的问题，顺势列出指标；
6. 仿真、训练设置。

#pagebreak()
== [-] RoboDuet (2024)

=== Introduction

提到了 Deep WBC，认为其开拓性地实现了身体和臂的协同，但未能解决末端位姿的*精确追踪*。废话有点多，总之是提出一个两阶段训练法和协作接口，先训练 locomotion 策略，再训练 arm 的并且可以和 locomotion 协作的策略，详见方法节。

=== Method

*Loco 策略* $pi_"loco"$ 目标是跟踪目标指令 $bold(c)_t = (v_x^"cmd", v_y^"cmd", omega_"yaw"^"cmd", phi.alt_"pitch"^"cmd", phi.alt_"roll"^"cmd")$，分别是 base 坐标系下目标线速度 $v_(x, y)^"cmd" = (v_x^"cmd", v_y^"cmd")$、目标偏航角速度 $omega_"yaw"^"cmd"$ 和目标姿态角 $bold(phi.alt)^"cmd" = (phi.alt_"pitch"^"cmd", phi.alt_"roll"^"cmd")$。Loco 策略的观测 $o_t^"loco"$ 包括当前腿状态 $s_t^"leg" in RR^24$（四条三关节腿的关节位置和关节角速度）、身体状态 $bold(phi.alt)_t$（roll 和 pitch 角）、前述目标指令 $bold(c)_t$、时间 $bold(t)_t$ 和上一步腿部动作 $a_(t-1)^"leg"$。

*Arm 策略* $pi_"arm"$ 的目标是精确地追踪 6D 末端位姿。其观测 $o_t^"arm"$ 包括臂状态 $s_t^"arm" in RR^12$（六自由度关节角和关节速度）、目标末端位姿 $chi_t in RR^9$、*身体状态* $bold(phi.alt)_t$、上一步臂动作 $a_(t-1)^"arm" in RR^8$。Arm 策略的动作空间包含臂动作 $a_t^("arm"^J) in RR^6$ 和用于替代 $bold(phi.alt)^"cmd"$ 的 $a_t^("arm"^G) = (a_t^("arm"^G_p), a_t^("arm"^G_r)) in RR^2$，为臂控制提供身体多给的两个自由度。然后 Arm 策略的动作 $a_t^("arm"^J) in RR^6$ 会和 Loco 策略输出的动作拼接以控制整个系统。

训练采用*两阶段训练*方案，一阶段将机械臂形态锁死、目标设为零，只训练 Loco 策略，引用 _Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior_ 的 MoB，详见对应章节，但固定部分行为参数以加快训练和适配协作框架（例如身体高度）。还加上一项瞬时能量抑制的惩罚以减轻臂重量和惯性对腿部电机的压力。

二阶段……

=== Remarks

不写了，感觉没什么价值。*本文主要是*架构上 Arm 策略顺手给 Loco 策略发指令比较新奇，其他主要都是围绕这个框架的实现进行的各种努力，效果上较其他工作略有提升但幅度不大，也不能算实现了 accurate pose tracking。

#pagebreak()
== [++] Walk These Ways (2022)

早期高引工作。文章指出平地速度指令可以由各种不同的步态执行，不同步态也可能适应不同的现实场景；提出 Multiplicity of Behavior（MoB），为策略添加一个行为参数输入 $bold(b)_t$，训练时根据这些参数调整奖励设计，最终所得策略 $pi (dot mid(|) bold(c)_t, bold(b)_t)$ 就可调节了（$bold(c)_t$ 是原有的 command）。

到这里重点就是三个问题，第一是如何设计全面的、符合直觉的行为参数 $bold(b)_t$ 定义，第二是如何设计奖励实现参数数值与期望的物理行为之间的对应，第三是训练如何进行。

*行为参数* $bold(b)_t = [bold(theta)_1^"cmd", bold(theta)_2^"cmd", bold(theta)_3^"cmd", bold(f)^"cmd", bold(h)_z^"cmd", bold(phi.alt)^"cmd", bold(s)_y^"cmd", bold(h)_z^(f,"cmd")]$ 具体包括：
1. $bold(theta)^"cmd" = (bold(theta)_1^"cmd", bold(theta)_2^"cmd", bold(theta)_3^"cmd")$ 是足相位参数，可表示例如 pronking $(0, 0, 0)$、trotting $(0.5, 0, 0)$、bounding $(0, 0.5, 0)$、pacing $(0, 0, 0.5)$ 以及它们的连续插值例如 galloping $(0.25, 0, 0)$，总之可以表示所有 two-beat quadrupedal contact patterns；
2. $bold(f)^"cmd"$ 是步频，单位 Hz，即每条腿每秒内接触地面次数；
3. $bold(h)_z^"cmd"$ 是身体高度指令；
4. $bold(phi.alt)^"cmd"$ 是身体 pitch 指令；
5. $bold(s)_y^"cmd"$ 是足间距（宽）指令；
6. $bold(h)_z^(f,"cmd")$ 是抬腿高度指令。

然后是*奖励设计*，详见文章 Table 1。主要是分为 task（追踪原始速度指令的奖励）、augmented auxiliary（行为参数 $bold(b)_t$ 控制的辅助奖励）和 fixed auxiliary（与 $bold(b)_t$ 无关用于维持基本稳定的辅助奖励）几部分。基本的追踪、摆荡抑制等不说，行为参数中例如步频、相位等是通过构造各足期望接触时序，然后在期望抬起时惩罚受力、期望接触时惩罚移动实现的。

文章还希望可以丝滑地在线调整行为参数，也就是第三个问题，*训练*时应该要涵盖参数中途切换的 episodes。还有采样、策略输入、网络架构等细节略去。

=== Remarks

添加到策略输入的参数、行为参数的设计、奖励的设计。感觉重点在训练上，考虑复现一下。

#pagebreak()
== [] Visual WBC (2024)

TODO

=== Remarks

TODO

#pagebreak()
== [++] UMI on Legs (2024)

=== Introduction

使用机器人真机数据采集成本高昂且难以迁移，仿真则是多样化的任务、奖励设置成为瓶颈。文章提出 UMI-on-Legs：
1. 现实世界中无需机器人即可扩展 task-centric 数据：用来自 UMI 的通用夹爪采集数据，用 diffusion policy 建模和除末端执行器之外机器人本体无关的策略；
2. 仿真中无需任务设置即可扩展 robot-centric 数据：大规模并行训练追踪末端执行器轨迹的 WBC。

=== Method

说白了就是一个低速操作策略给出未来一段时间内的相对（相机系）末端轨迹，然后一个高速 WBC 追踪这个末端轨迹，很干净。

操作策略用的是 a U-Net architecture diffusion policy with the DDIM scheduler and a pre-trained CLIP vision encoder，动作块长度 64。

*全身控制器*是重点，文章先提了一些 prior work：Deep WBC、_Learning Force Control for Legged Manipulation_、RoboDuet 和 Visual WBC，认为它们从身体坐标系采样目标位姿训练，无法抵抗过程中的扰动（就是中途移动身体位置时它们不会让末端执行器停在原地，而本文希望如此）。文章希望控制器在 task-frame 中追踪末端位姿。

观测空间是腿和臂的 18 个关节的位置和速度、身体朝向和角速度、上一时刻动作、目标末端轨迹指令（3D position + 6D rotation）。目标轨迹*采样*相对当前时刻 ${-60, -40, -20, 0, 20, 40, 60, 1000}$ ms 的样本，用以指示近时刻和远时刻的目标以及当前的速度/加速度情况。

*奖励*设计在附录中，包括：
1. joint limit、joint acceleration、joint torque、root height、collision、action rate 这些常用项；
2. *身体-末端对齐*：正则化机械臂的 0、3 号关节（影响 yaw 的主要关节）位置，让末端自然和身体对齐，值得记录；
3. 均匀质量分布：正则化四腿支持力标准差令其受力均匀；
4. 脚髋对齐（feet under hips）：正则化各脚与其对应髋关节平面投影的差，提升站立姿态稳定性；
5. 末端位姿：关键的让末端执行器到达目标位姿的奖励，统一 position 和 orientation 项使用 $exp(-(epsilon.alt_"pos"/sigma_"pos" + epsilon.alt_"orn"/sigma_"orn"))$，$epsilon.alt$ 为 error，$sigma$ 为课程（curriculum）系数，随 error 降低提高以让奖励函数更尖锐、引导更强。

策略网络直接就是一个观测到 $12 + 6$ 动作空间（关节位置）的 MLP，关节位控由 PD 控制力矩实现。



=== Remarks

操作策略只给出末端位姿轨迹，其他都训练 WBC 搞定的思路值得参考，不知道复现会不会存在很多问题，换操作策略又如何，值得尝试。

也需要重点关注 WBC 部分的观测、动作、奖励设计，尤其奖励中有几个优化姿态的项。

TODO 还有一部分关于 imu 什么的讨论
