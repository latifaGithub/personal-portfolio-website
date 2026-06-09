---
layout: post
type: paper-review
title: "Notes on: Attention Is All You Need"
paper_title: "Attention Is All You Need"
authors: "Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin"
venue: "NeurIPS 2017"
year: 2017
paper_url: "https://arxiv.org/abs/1706.03762"
tags: [NLP, Transformers, Deep Learning, Attention]
description: "My notes on the paper that introduced the Transformer architecture — the foundation of every modern large language model, and why self-attention was such a fundamental departure from recurrence."
cover: "linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%)"
cover_icon: "🤖"
---

Before reading this paper I had already used transformers in projects — BERT for text classification, GPT-2 for generation. But using a tool and understanding it are different things. Reading the original paper changed how I think about sequence modelling at a fundamental level.

---

## What the Paper Proposes

Vaswani et al. replace recurrent and convolutional layers entirely with a mechanism called **self-attention**. Their claim is direct: attention alone is sufficient to achieve state-of-the-art machine translation, and doing so makes the model dramatically faster to train.

The architecture has two main components:

- **Encoder** — processes the full input sequence and produces a rich contextual representation of every token
- **Decoder** — generates the output sequence one token at a time, attending to both the encoder output and previously generated tokens

Both are built entirely from stacked self-attention and feed-forward layers. There is no recurrence anywhere.

---

## The Core Mechanism: Scaled Dot-Product Attention

The fundamental operation is:

**Attention(Q, K, V) = softmax( QKᵀ / √dₖ ) · V**

Every token computes a query, a key, and a value. The query of one token is compared (via dot product) against the keys of every other token to produce attention weights — essentially, how much each token should attend to each other token. Those weights are then used to form a weighted sum of the values.

The scaling by √dₖ is a detail I almost overlooked on first read, but it matters: without it, the dot products grow large for high-dimensional spaces and push the softmax into regions with near-zero gradients, stalling learning.

**Why this is better than recurrence:** In an RNN, the hidden state at step t depends on step t−1, so you cannot process the sequence in parallel. With self-attention, every position attends to every other position simultaneously. This is the primary reason transformers scale so well on modern GPUs.

---

## Multi-Head Attention

Rather than running one attention function, the paper uses h attention heads in parallel, each operating in a different learned subspace of the input. The outputs are concatenated and projected:

**MultiHead(Q, K, V) = Concat(head₁, …, headₕ) · Wᴼ**

My intuition for why this helps: different heads can specialise. One head might track syntactic relationships (subject-verb agreement), another might capture coreference, another might attend to positional neighbours. Later interpretability work by Clark et al. (2019) confirmed that heads do in fact develop distinct, interpretable attention patterns.

---

## Positional Encoding

Self-attention is **permutation-invariant** — it has no built-in notion of order. If you shuffle the words in a sentence, the attention weights just rearrange accordingly and the output is the same. That is obviously wrong for language.

The solution is to inject positional information by adding sinusoidal positional encodings to the input embeddings:

- PE(pos, 2i) = sin(pos / 10000^(2i/dmodel))
- PE(pos, 2i+1) = cos(pos / 10000^(2i/dmodel))

What I found interesting is that this is not learned — it is fixed. The authors argue that sinusoidal encodings allow the model to generalise to sequence lengths longer than those seen during training, because any fixed offset can be represented as a linear combination of the encodings. Later work (BERT, GPT) switched to learned positional embeddings, but the sinusoidal approach holds up well.

---

## What I Think About the Paper

**What is genuinely impressive:** The central insight is almost retrospectively obvious — replace sequential computation with parallel attention — but it took years for anyone to articulate it this cleanly and validate it at this scale. The ablation study in Table 3 is particularly convincing: it strips away components one at a time, showing that each design choice is load-bearing.

**What I would have pushed further:** The paper establishes that attention *works* empirically, but the theoretical question of *why* attention is such a good inductive bias for language is largely left open. That gap has generated a large and fascinating interpretability literature, which I now actively follow.

**A subtlety that matters in practice:** The paper uses **layer normalisation before** the residual addition (in some formulations) and **after** in others. This "pre-LN vs. post-LN" distinction turns out to significantly affect training stability for very deep transformers. The original paper uses post-LN; later work found pre-LN easier to train at scale.

**Connection to my research interests:** The attention mechanism has spread far beyond NLP — it now appears in vision (Vision Transformers), graphs (Graph Attention Networks), and multimodal models. Understanding the original formulation gives me a much clearer mental model when I encounter these variants.

---

## Key Takeaways

- Self-attention allows every token to attend to every other token simultaneously, replacing the sequential bottleneck of RNNs
- The scaling factor √dₖ prevents gradient vanishing in the softmax — a small detail with large practical consequences
- Multi-head attention allows the model to jointly attend to different types of relationships in parallel subspaces
- Sinusoidal positional encodings inject order information without requiring learned parameters
- The parallelism of the architecture, not just its accuracy, is what made it take over the field — it trains orders of magnitude faster than LSTMs at scale

---

## What to Read Next

- **BERT (Devlin et al., 2019)** — encoder-only transformer pre-trained with masked language modelling; how transformers became general-purpose
- **GPT-3 (Brown et al., 2020)** — what happens when you scale a decoder-only transformer to 175B parameters
- **An Image is Worth 16×16 Words (Dosovitskiy et al., 2021)** — the transformer applied to vision patches
- **In-context Heads and Attention Patterns (Clark et al., 2019)** — what the individual attention heads actually learn
