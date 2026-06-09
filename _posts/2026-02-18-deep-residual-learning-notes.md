---
layout: post
type: paper-review
title: "Notes on: Deep Residual Learning for Image Recognition"
paper_title: "Deep Residual Learning for Image Recognition"
authors: "He, Zhang, Ren, Sun"
venue: "CVPR 2016"
year: 2016
paper_url: "https://arxiv.org/abs/1512.03385"
tags: [Computer Vision, Deep Learning, CNNs, Architecture]
description: "My notes on the ResNet paper — the insight that made training extremely deep neural networks possible, and why skip connections are one of the most impactful ideas in modern deep learning."
cover: "linear-gradient(135deg, #0d1b2a 0%, #1a3a2a 50%, #0a1f15 100%)"
cover_icon: "🧠"
---

Before ResNet, there was a strange problem in deep learning: adding more layers to a neural network did not always help — it often made things worse. Not just on the test set, where you might blame overfitting, but on the *training* set too. This shouldn't happen in theory. A deeper network should be able to learn at least the same function as a shallower one, simply by setting the extra layers to identity mappings. But in practice, optimisation broke down.

He et al. named this the **degradation problem**, and proposed a deceptively simple fix.

---

## The Core Idea: Residual Connections

Instead of asking a stack of layers to learn a mapping **H(x)** directly, let them learn the *residual* **F(x) = H(x) − x**. The original input is added back via a shortcut connection:

**output = F(x) + x**

This shortcut adds no parameters and no extra computation (when dimensions match). But it fundamentally reframes what the optimiser has to find. If the right answer is close to the identity — do nothing, just pass the input through — then the residual should be close to zero, and learning zero is easy. Plain networks have no such shortcut; they have to learn the identity from scratch through a stack of nonlinear layers, which is hard.

---

## Why This Fixes the Degradation Problem

My understanding comes down to gradient flow. In a plain deep network, the gradient signal degrades as it propagates backward through many weight matrices and activation functions. With skip connections, the gradient has a **direct highway** back to earlier layers, bypassing the weight matrices in between. This keeps the signal strong throughout training.

There is also a beautiful way to interpret a ResNet architecturally: unrolling the skip connections reveals an implicit **ensemble of paths** of varying depths. Short paths carry gradients directly and dominate early training; deeper paths learn more complex transformations as training progresses. Veit et al. (2016) formalised this interpretation and it changed how I think about the architecture — less as a single deep network, more as a collection of shallow networks sharing weights.

---

## The Experimental Evidence

The paper's key comparison is between plain networks and ResNets at 18 and 34 layers, trained on ImageNet.

**Plain networks:** 34-layer performs *worse* than 18-layer on the training set. More capacity, worse results. This is the degradation problem in action.

**ResNets:** 34-layer performs *better* than 18-layer. The skip connections flip the direction of the depth-performance relationship.

The paper then scales to 152 layers — far deeper than anything successfully trained before — and achieves 3.57% top-5 error on ImageNet, winning ILSVRC 2015. At the time, this was better than the reported human-level performance on this benchmark.

What I found most convincing is that the improvement is not marginal. The 152-layer ResNet doesn't just beat the 34-layer by a small margin — it improves substantially. Depth was genuinely being unlocked, not just added.

---

## Bottleneck Architecture

For networks deeper than 50 layers, the paper introduces a **bottleneck design**: a 1×1 convolution reduces dimensionality, a 3×3 convolution processes it, and another 1×1 convolution restores dimensionality. This keeps each residual block computationally tractable while preserving representational capacity.

I missed the significance of this on first read and treated it as an implementation detail. It isn't. The bottleneck is what makes ResNet-101 and ResNet-152 practical to train, and it's the direct ancestor of the bottleneck blocks used in MobileNets, EfficientNets, and transformer feedforward layers.

---

## What I Think About the Paper

**What is compelling:** The problem is clearly defined and the solution is clearly motivated. The paper doesn't just show that ResNets work — it shows *why* they're needed (the degradation experiments), *what* makes them different (the residual formulation), and *how far* they scale (152 layers, competition-winning results). The logic is tight.

**What I wish had been explored:** The paper focuses on image classification. The broader question of when skip connections are necessary — at what depth does degradation set in, and does it depend on the task — is left open. In practice, ResNet backbones got applied to detection, segmentation, and generation almost immediately, which suggests the benefit generalises widely, but I would have liked some systematic analysis of that.

**A practical lesson:** I used to add ResNet backbones to vision projects almost automatically, without thinking about whether they were necessary. Reading this paper made me more deliberate. Skip connections are most valuable when you need depth and face optimisation difficulty. For shallow networks on simple tasks, a plain network is often fine.

**Connection to my work:** ResNet features are the starting point for most of the transfer learning I do in computer vision. Understanding why the architecture works — not just that it works — makes me a better debugger when fine-tuning breaks down or when I need to decide how many layers to unfreeze.

---

## Key Takeaways

- Deep plain networks suffer degradation: training accuracy gets *worse* with more layers due to optimisation failure, not overfitting
- Skip connections provide a direct gradient highway to early layers, solving the degradation problem
- Residual learning reframes the task: layers learn what to *add* to the input, not a full transformation from scratch
- The bottleneck design (1×1 → 3×3 → 1×1) makes very deep networks computationally feasible
- ResNet backbones became the default for nearly all of computer vision — detection, segmentation, face recognition, medical imaging — within two years of this paper

---

## What to Read Next

- **Identity Mappings in Deep Residual Networks (He et al., 2016)** — the follow-up that analyses *where* to place batch norm and activation for best results
- **Residual Networks Behave Like Ensembles of Relatively Shallow Networks (Veit et al., 2016)** — the ensemble interpretation of skip connections
- **DenseNet (Huang et al., 2017)** — extends the skip-connection idea: every layer connects to every subsequent layer
- **EfficientNet (Tan & Le, 2019)** — principled compound scaling of depth, width, and resolution, building on residual insights
