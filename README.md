# AutoFuse-GPU-Optimizer
A learning-based GPU compiler optimizer using GNNs, Transformers, and RL to automate kernel fusion and scheduling for faster deep learning inference.

# AutoFuse-MLCompiler

*A Deep Learning–Driven GPU Compiler Optimizer*

## Overview

**AutoFuse-MLCompiler** is a learning-based GPU compiler optimization system that automatically improves deep learning inference performance. Instead of relying on hand-written heuristics or exhaustive auto-tuning, AutoFuse uses **deep learning models** to predict kernel performance, learn operator fusion strategies, and optimize memory layouts during compilation.

The project integrates **Graph Neural Networks (GNNs)**, **Transformer-based decision models**, and **Reinforcement Learning (RL)** into the compiler pipeline to deliver hardware-aware, scalable, and generalizable optimizations for modern NVIDIA GPUs.

---

## Motivation

Optimizing GPU kernels for deep learning workloads is increasingly difficult due to:

* Rapidly growing hardware complexity (tensor cores, memory hierarchies)
* Massive combinatorial optimization spaces
* Hardware-specific performance behavior
* High engineering cost of manual tuning

Suboptimal kernel implementations can lead to **2–10× performance loss**, increased
