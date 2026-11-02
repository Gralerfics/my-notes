#import "../generic.typ": *

#import "@preview/cetz:0.4.2"
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

= Large (Language) Model

== Transformer

https://arxiv.org/pdf/1706.03762

https://v11enp9ok1h.feishu.cn/wiki/RbIpw8NJHijTJXkQAqncTmRunBQ

=== Embedding

词嵌入，将词元通过预训练模型映射为特定长度 $d_v$ 的向量 $z_i in RR^(d_v)$；位置信息嵌入，将位置信息映射为同样长度的向量 $p_i in RR^(d_v)$。在 Transformer 中是将二者直接相加得到最终的词向量 $x_i in RR^(d_v)$。

将 $n$ 个词向量转置并纵向堆叠得到向量矩阵 $X = [x_1, x_2, dots, x_n]^top in RR^(n times d_v)$，

=== Self-Attention

通过可学习的权重矩阵 $W_q, W_k in RR^(d_v times d_k), W_v in RR^(d_v times d_v)$ 将词向量矩阵线性变换为 $Q, K in RR^(n times d_k), V in RR^(n times d_v)$ 矩阵：

$
Q = X W_q, quad K = X W_k, quad V = X W_v
$

接下来计算每个词向量之间的相似度，将每个词的 $q_i$（query）和每个词的 $k_j$（key）作点积得到 $Q K^top = [a_(i j)]$。之后我们希望将它的行当作权重来线性组合 $v_k$，即从第 $i$ 个词视角，按照其他词与之的相似度大小加权，把（经过 $V$ 变换的）词向量加到一起。为保证一致性，我们对每一行进行 softmax 运算将其转化为合法的概率分布；同时由于 $a_(i j)$ 数值受 $Q, K$ 维度的影响，在数值较大时会导致 softmax 梯度较小，于是我们在其内除以 $sqrt(d_k)$，最终有：

$
"Attention"(Q, K, V) = "softmax"((Q K^T) / sqrt(d)) V
$

对于该注意力运算，输入都是 $n times d_v$ 的矩阵，输出维度不变依旧是 $n times d_v$，最终每行是 $a_i = w_11 v_1 + w_12 v_2 + dots + w_(1n) v_n$，$w_(i j)$ 用以示意经过归一化和 softmax 后的 $a_(i j)$。

=== Multi-Head Self-Attention

多头注意力机制提供多组注意力计算提供学习多种相似度关系的可能。分为 $h$ 个头，运算过程为：

$
"MultiHead"(Q, K, V) = "Concat"("head"_1, dots, "head"_h) W^O \
"where" "head"_i = "Attention"(Q W_i^Q, K W_i^K, V W_i^V)
$

具体就是在计算注意力前添加 $W_i^Q, W_i^K in RR^(d_"model" times d_k), W_i^V in RR^(d_"model" times d_v)$ 以及注意力后拼接并添加 $W^O in RR^(h d_v times d_"model")$ 进行线性变换。这里的输入 $Q, K, V$ 就不是前面的维度了，竖着切成 $h$ 份后每份才是 $n times d_k$ 或 $n times d_v$。

=== Architecture

#Cre("TODO")

#blockquote[
    $Q$、$K$、$V$ 分别被称为 query、key 和 value，可以抽象地理解为 $Q$ 是发起读取的一方，$K$ 和 $V$ 是被读取的 memory。$Q K^T$ 表达应该从 memory 中查询哪些位置，再和 $V$ 相乘得到具体内容。
    
    例如在上述架构的 encoder-decoder cross-attention 中，encoder 输出的信息编码矩阵 $C$ 被投影到 $K$ 和 $V$ 送入 decoder，decoder 从前一层多头注意力输出 $Q$ 意在查询 “为了生成当前位置的 token，应该从 encoder 中读取什么信息”。
]

=== Training

== Tuning

=== Full Fine-Tuning

=== Low-Rank Adaptation (LoRA)

原本模型记为 $y = W x$，LoRA 冻结原参数 $W$，在其上加一个矩阵 $Delta W = A B$ 作为微调参数，其中 $A in RR^(r times d_"input"), B in RR^(d_"output" times r)$，$r$ 通常较小（低秩），使得微调时参数量也更小。

=== PEFT and `peft`

Hugging Face peft 库

== Other

=== Classifier-Free Guidance (CFG)


