---
title: PyTorch Implementation of Dependency Parsing
mathjax: true
date: 2026-01-16 23:26:07
categories: Course Notes
description: A more efficient and elegant PyTorch implementation of Dependency Parsing, optimized for GPU acceleration.
---

## Prerequisites

Wikipedia: [Dependency grammar](https://en.wikipedia.org/wiki/Dependency_grammar)

Assignment: [Assignment: Dependency parsing](https://github.com/coli-saar/cl/wiki/Assignment:-Dependency-parsing)

> In a nutshell, you will proceed as follows:
>
> - From the XLM-RoBERTa embeddings of each token, extract representations `H_head` and `H_dep` using a one-layer MLP with some output dimension $d$ (see the D&M paper for suggestions on hyperparameters). Note that you need a separate MLP for the head and for the dep representation.
>
> - Calculate a score for each pair of a potential head $i$ and potential dependent $j$, by multiplying `H_head[i].T * U1 * H_dep[j] + H_head[i].T * u2`. `U1` is a $d \times d$ matrix, and `u2` is a $d$-dimensional vector; their entries are parameters of the model which are learned in training.

## Head Prediction

The original formula is: $S = (H^{(hd)})^T \cdot U_1 \cdot H^{(dep)} + (H^{(hd)})^T \cdot u_2$, where $H^{(hd)} \in\mathbb{R}^{d \times n}$, $H^{(dep)} \in\mathbb{R}^{d \times m}$, $U_1 \in\mathbb{R}^{d \times d}$, $u_2 \in\mathbb{R}^{d \times 1}$, $S \in\mathbb{R}^{n \times m}$ ($n=m$ is the number of words and $d$ is a hyperparameter), and $S_{i,j}$ is the score that $i$ is the head and $j$ is the dependent.

With the assumption that every token must have exactly one head (except the root which has none), we would theoretically apply `softmax` along dimension $i$ of the **logits** to obtain the probability score $S$. THowever, this deviates from standard PyTorch conventions, particularly concerning `CrossEntropyLoss`, which excepts `softmax` to be applied along the last dimension by default. To resolve this, I refactored the formula to calculate the transpose, $S^T$. Consequently, $(S^T)_{i,j}$ represents the score of $i$ being the dependent and $j$ beging the head,allowing us to seamlessly apply `softmax` to the last dimension.

$$
\begin{aligned}
S^T &= ((H^{(hd)})^T \cdot U_1 \cdot H^{(dep)} + (H^{(hd)})^T \cdot u_2)^T \\\\
    &= ((H^{(hd)})^T \cdot U_1 \cdot H^{(dep)})^T + ((H^{(hd)})^T \cdot u_2)^T \\\\
    &= (H^{(dep)})^T \cdot U_1^T \cdot H^{(hd)} + u_2^T \cdot H^{(hd)} \\\\
    &= ((H^{(dep)})^T \cdot U_1^T + u_2^T) \cdot H^{(hd)}
\end{aligned}
$$

Consider the term $(H^{(dep)})^T \cdot U_1^T + u_2^T$, where $U_1^T \in\mathbb{R}^{d \times d}$, $u_2^T \in\mathbb{R}^{1 \times d}$, and $(H^{(dep)})^T \in\mathbb{R}^{n \times d}$. Recall that the standard definition of a neural network linear layer is $\text{LinearLayer}(X\ |\ W, b) = X \cdot W + b$ where $W \in\mathbb{R}^{d \times d}$, $b \in\mathbb{R}^{1 \times d}$, and $X \in\mathbb{R}^{n \times d}$. By comparing the two, it becomes evident that this term is mathematically equivalent to a standard linear layer. Here, $U_1^T \in\mathbb{R}^{d \times d}$ serves as the weight matrix, $u_2^T \in\mathbb{R}^{1 \times d}$ acts as the bias, and $(H^{(dep)})^T \in\mathbb{R}^{n \times d}$ is the input. Thus, the equation simplifies to $S^T = \text{LinearLayer}((H^{(dep)})^T\ |\ U_1^T, u_2^T) \cdot H^{(hd)}$

**Formulating this operation as a native PyTorch Linear layer not only results in cleaner code, but also allows PyTorch to leverage low-level optimizations (such as Tensor Cores) for significantly higher GPU utilization.**

## Edge Label Prediction

The original formula is: $S_e = (H^{(hd-e)})^T \cdot W \cdot H^{(dep-e)} + U_3 \cdot H^{(hd-e)} + U_4 \cdot H^{(dep-e)} + u_5$, in which $H^{(hd-e)} \in\mathbb{R}^{d \times n}$, $H^{(dep-e)} \in\mathbb{R}^{d \times m}$, $W \in\mathbb{R}^{d \times c \times d}$, $U_3, U_4 \in\mathbb{R}^{c \times d}$, $u_5 \in\mathbb{R}^{c \times 1}$, $S \in\mathbb{R}^{n \times c \times m}$, $c$ is the number of edge classes.

For a given dependent $i$ with head $j$, we exclusively need the score $S_{ji}$. Because $i$ and $j$ are coupled during this phase, we can eliminate the head dimension $j$ entirely. We achieve this by gathering the features of the predicted heads, effectively transforming the matrix $(H^{(hd-e)})^T \in\mathbb{R}^{n \times d}$ into tensor of **batched vectors**, $(h^{(hd-e)})^T \in\mathbb{R}^{1 \times d \times m}$. Consequently, the computational complexity is drastically reduced from $O(n^2)$ to $O(n)$.

$$
\begin{aligned}
S_e^T &= ((h^{(hd-e)})^T \cdot W \cdot H^{(dep-e)} + U_3 \cdot h^{(hd-e)} + U_4 \cdot H^{(dep-e)} + u_5)^T \\\\
      &= ((h^{(hd-e)})^T \cdot W \cdot H^{(dep-e)})^T + (U_3 \cdot h^{(hd-e)} + U_4 \cdot H^{(dep-e)} + u_5)^T \\\\
      &= (H^{(dep-e)})^T \cdot W^T \cdot h^{(hd-e)} + (h^{(hd-e)})^T \cdot U_3^T + (H^{(dep-e)})^T \cdot U_4^T + u_5^T \\\\
\end{aligned}
$$

Let us examine the term $(h^{(hd-e)})^T \cdot U_3^T + (H^{(dep-e)})^T \cdot U_4^T + u_5^T$. By concatenating $U_3, U_4$, we form a new weight matrix $U_{34} \in\mathbb{R}^{R \times 2d}$. Similarly, by concatenating $h^{(hd-e)}$ and $H^{(dep-e)}$, we construct a joint feature matrix $H^{(hd-dep-e)} \in\mathbb{R}^{2d \times m}$. The expression is then equivalent to $(H^{(hd-dep-e)})^T \cdot U_{34}^T + u_5^T$. Once again, this structure perfectly aligns with a standard linear layer. Therefore, the final streamlined equation is: $S_e^T = (H^{(dep-e)})^T \cdot W^T \cdot h^{(hd-e)} + \text{LinearLayer}(\text{cat}(h^{(hd-e)}, H^{(dep-e)})^T\ |\ U_{34}^T, u_5^T)$

## Coding

```python
@torch.no_grad
def word_cle(
    logits_arc: FP[T, "batch dependent=token head=token"],
    word_first_token_mask: Bool[T, "batch dependent=token"],
) -> Int[T, "batch dependent=token"]:
    """Calculate the MST of words using Chu-Liu-Edmonds algorithm.

    Args:
        logits_arc: `FP[T, "batch dependent=token head=token"]` Logits of the arc prediction.
        word_first_token: `Bool[T, "batch token"]` Index mask of the first tokens of the words.

    Returns:
        `Int[T, "batch dependent=token]` Predicted arcs corresponding to each word, non-first token would be set as 0.
    """
    assert not torch.any(word_first_token_mask[:, 0]), "index 0 must be the root"
    results = torch.zeros_like(word_first_token_mask, device="cpu", dtype=int)
    word_first_token_mask = word_first_token_mask.to(device="cpu", copy=True).numpy(),
    word_first_token_mask[:, 0] = True  # fetch the root for CLE
    for logits, mask, res in zip(
        logits_arc.to(device="cpu", dtype=torch.float64).numpy(),
        word_first_token_mask,
        results,
    ):
        word_logits = logits[mask][:, mask]  # T[1+word, 1+word]
        # log-softmax is not need here. Check the Calculation Note in README
        word_preds, _ = chu_liu_edmonds(word_logits)  # T[1+word]
        # map words to their corresponding first tokens.
        wordid_to_first_tokenid = np.where(mask)[0]
        res[mask] = torch.from_numpy(wordid_to_first_tokenid[word_preds[1:]])
    # word_first_token_mask[:, 0] = False
    return results

class RobertaDependencyParser(RobertaDependencyParserBase):
    def __init__(
        self,
        backbone: XLMRobertaModel,
        padding_label: int,
        training_backbone: bool,
        dim_h_arc: int,
    ):
        super().__init__(backbone, padding_label, training_backbone)
        self.mlp_h_arc_head = Mlp([self._dim_backbone, dim_h_arc])
        self.mlp_h_arc_dep = Mlp([self._dim_backbone, dim_h_arc])
        # weight as U1, bias as u2, check the Calculation Notes in README.
        self.linear_u12 = torch.nn.Linear(dim_h_arc, dim_h_arc)

        self.mlp_h_rel_head = Mlp([self._dim_backbone, dim_h_rel])
        self.mlp_h_rel_dep = Mlp([self._dim_backbone, dim_h_rel])
        # weight as [U3,U4], bias as u5, check the Calculation Notes in README.
        self.linear_u345 = torch.nn.Linear(dim_h_rel * 2, num_edge_classes)
        self.mat_w = torch.nn.Parameter(
            torch.randn(num_edge_classes, dim_h_rel, dim_h_rel)
        )

    def _forward_arc(
        self,
        backbone_out: FP[T, "batch token _dim_backbone"],
        heads: Int[T, "batch dependent=token"],
    ) -> tuple[
        FP[T, ""],
        FP[T, "batch dependent=token head=token"],
        Int[T, "batch dependent=token"],
    ]:
        """
        Returns:
            A tuple contains
            - `FP[T, ""]` Cross entropy loss of arc prediction.
            - `FP[T, "batch dependent=token head=token"]` Logits of the arc prediction.
            - `Int[T, "batch dependent=token"]` Predicted arcs.
        """
        BATCH, TOKEN, _ = backbone_out.shape

        # MLP on the last dim
        h_head: FP[T, "batch head dim_h_arc"] = self.mlp_h_arc_head(backbone_out)
        h_dep: FP[T, "batch dependent dim_h_arc"] = self.mlp_h_arc_dep(backbone_out)

        term: FP[T, "batch dependent dim_h_arc"] = self.linear_u12(h_dep)
        logits: FP[T, "batch dependent head"] = torch.einsum(
            "bid,bjd->bij", term, h_head
        )
        arcs = word_cle(logits, heads != self._padding_label).to(heads.device)

        # calculate cross entropy loss
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(BATCH * TOKEN, TOKEN),
            heads.reshape(BATCH * TOKEN),
            ignore_index=self._padding_label,
        )
        return loss, logits, arcs

    def forward(
        self,
        input_ids: Int[T, "batch token"],
        attention_mask: Int[T, "batch token"],
        heads: Int[T, "batch dependent=token"],
        deprels: Int[T, "batch dependent=token"],
        **_,
    ) -> tuple[FP[T, ""], dict[str, T]]:
        r"""Dependency Parsing based on RoBERTa, with edge label prediction.

        Args:
            input_ids: `Int[T, "batch token"]` Batched tokens indices,
            attention_mask: `Int[T, "batch token"]` Attention mask passed to the backbone,
            heads: `Int[T, "batch dependent=token"]` Ground truth to calculate the loss.

        Returns:
            A tuple contains
            - `FP[T, ""]` Loss $L_{arc} + L_{rel}$.
            - Named outputs as a dict
                - logits_arc: `FP[T, "batch dependent=token head=token"]` Logits of the arc prediction.
                - logits_rel: `FP[T, "batch dependent=token num_edge_classes"]` Logits of the rel prediction.
                - arcs: `Int[T, "batch dependent=token"]` Predicted arcs.
        """
        # heads and deprels must be both provided or both not provided
        assert not ((heads is None) ^ (deprels is None))
        BATCH, TOKEN = input_ids.shape
        backbone_out: FP[T, "batch token self._dim_backbone"] = self.backbone.forward(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

        # predict arcs and calculate L_{arc} firstly
        loss, logits_arc, arcs = self._forward_arc(backbone_out, heads)
        pred_heads: Int["batch dependent"] = arcs
        if self.training:  # use ground truth in training.
            pred_heads = heads.clone()
            pred_heads[heads == self._padding_label] = 0

        # MLP on the last dim, get rel features
        h_rel_dep: FP[T, "batch dependent dim_h_rel"] = self.mlp_h_rel_dep(backbone_out)
        h_rel_head: FP[T, "batch head dim_h_rel"] = self.mlp_h_rel_head(backbone_out)
        # only select feature vectors from the predicted head.
        h_rel_head: FP[T, "batch dependent dim_h_rel"] = torch.gather(
            h_rel_head, 1, pred_heads.reshape(BATCH, TOKEN, 1).expand_as(h_rel_head)
        )

        term_a: FP[T, "batch dependent num_edge_classes"] = torch.einsum(
            "bid,cde,bie->bic", h_rel_dep, self.mat_w, h_rel_head
        )
        term_b: FP[T, "batch dependent _"] = torch.cat([h_rel_dep, h_rel_head], -1)
        term_b: FP[T, "batch dependent num_edge_classes"] = self.linear_u345(term_b)
        logits_rel: FP[T, "batch dependent num_edge_classes"] = term_a + term_b

        # calculate cross entropy loss: L = L_{arc} + L_{rel}
        # ignore incorrect edges
        deprels = deprels.clone()
        deprels[pred_heads != heads] = self._padding_label
        if torch.any(deprels != self._padding_label):
            loss = loss + torch.nn.functional.cross_entropy(
                logits_rel.reshape(BATCH * TOKEN, -1),
                deprels.reshape(BATCH * TOKEN),
                ignore_index=self._padding_label,
            )

        return loss, {"logits_arc": logits_arc, "logits_rel": logits_rel, "arcs": arcs}
```
