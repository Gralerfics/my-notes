#import "../generic.typ": *

#import "@preview/codly:1.3.0": *

#show: codly-init.with()

= Policy Gradient

// $forall s in cal(S), forall f: cal(S) times cal(A) -> RR$

// $
// nabla_theta EE_(A~pi_theta (dot mid(|) s)) [f(s, A)] = EE_(A~pi_theta (dot mid(|) s)) [f(s, A) nabla_theta ln pi_theta (A mid(|) s)]
// $

// $
// nabla_theta EE_(pi_theta) [R_t mid(|) s_t, a_t] = &gamma EE_(pi_theta) [R_(t+1) nabla_theta ln pi_theta (A_(t+1) mid(|) S_(t+1)) mid(|) s_t, a_t] \
// &+ gamma EE_(pi_theta) [nabla_theta EE_(pi_theta) [R_(t+1) mid(|) S_(t+1), A_(t+1)] mid(|) s_t, a_t]
// $

// policy gradient theorem:

// $
// nabla_theta J^(pi_theta) = nabla_theta EE_(pi_theta) [R_0] = EE_(pi_theta) [sum_(t=0)^(n-1) gamma^t R_t nabla_theta ln pi_theta (A_t mid(|) S_t)]
// $

// TODO 等一下，上面这些大写 R 貌似不是 reward 是 return，作业和 slides 和其他什么的符号都乱的
