---
sidebar_position: 5
title: Machine Learning & AI
---

# Machine Learning & AI Examples

Natural language prompts for ML/AI training and experimentation.

## Language Models

```
Train a character-level LSTM on Shakespeare and generate sonnets

Build a GPT-2 style model with 4 layers and train on code

Create word embeddings using skip-gram on Wikipedia abstracts

Train a BERT-tiny for sentiment classification

Build a seq2seq model for simple translation (numbers to words)

Train an autoregressive model to complete Python functions

Create sentence embeddings using contrastive learning

Build a small T5 model for text summarization

Train a tokenizer using BPE on a custom corpus

Fine-tune embeddings for semantic similarity
```

## Transformers from Scratch

```
Implement multi-head attention and verify against PyTorch

Build a transformer encoder and train on text classification

Create positional encodings (sinusoidal and learned)

Implement the transformer decoder with causal masking

Train a vision transformer (ViT) on CIFAR-10

Build a BERT-style masked language model

Implement rotary position embeddings (RoPE)

Create a mixture-of-experts transformer layer

Train a sparse attention transformer

Implement flash attention and compare memory usage
```

## Computer Vision

```
Train ResNet-18 from scratch on CIFAR-10

Build a U-Net for image segmentation

Train an autoencoder to reconstruct MNIST digits

Create a GAN to generate faces

Train YOLO-style object detection on a custom dataset

Build a siamese network for one-shot learning

Train a neural style transfer model

Create a depth estimation network from single images

Train a pose estimation model for human keypoints

Build an image captioning model with attention
```

## Generative Models

```
Train a VAE on MNIST and interpolate in latent space

Build a diffusion model for image generation

Create a flow-based generative model (RealNVP style)

Train a GAN with spectral normalization

Build a VQ-VAE for discrete latent codes

Train an autoregressive image model (PixelCNN)

Create a conditional GAN for image-to-image translation

Build a neural ODE for continuous normalizing flows

Train a score-based generative model

Create a latent diffusion model for high-res images
```

## Reinforcement Learning

```
Train DQN to play Atari Breakout

Implement policy gradient (REINFORCE) for CartPole

Build an actor-critic agent for continuous control

Train PPO on MuJoCo environments

Implement curiosity-driven exploration

Build a model-based RL agent with world models

Train multi-agent RL for competitive games

Implement hindsight experience replay

Build an offline RL agent from logged data

Train an agent using human feedback (RLHF style)
```

## Neural Network Fundamentals

```
Visualize what each layer learns in a CNN

Compute and visualize attention weights in a transformer

Show gradient flow through a deep network

Demonstrate vanishing gradients in RNNs vs LSTMs

Visualize the loss landscape around optima

Compare batch norm, layer norm, and group norm

Show the effect of dropout at different rates

Visualize weight initialization strategies

Demonstrate mode collapse in GAN training

Show the lottery ticket hypothesis with pruning
```

## Optimization & Training

```
Compare Adam, SGD, and AdamW on the same model

Implement learning rate warmup and cosine annealing

Show the effect of batch size on convergence

Implement gradient clipping and show its effect

Compare different weight initialization methods

Implement mixed precision training

Show the effect of label smoothing

Implement early stopping with patience

Compare different data augmentation strategies

Implement gradient accumulation for large batches
```

## Graph Neural Networks

```
Build a GCN for node classification on Cora

Train a graph attention network (GAT)

Implement message passing neural networks

Build a model for molecular property prediction

Train a GNN for link prediction

Implement graph pooling for graph classification

Build a temporal graph network

Train a heterogeneous graph neural network

Implement over-smoothing analysis in deep GNNs

Build a knowledge graph embedding model
```

## Time Series & Sequences

```
Train an LSTM for stock price prediction

Build a temporal fusion transformer

Implement wavenet-style dilated convolutions

Train a neural ODE for irregular time series

Build an attention-based anomaly detector

Implement N-BEATS for time series forecasting

Train a transformer for multi-step prediction

Build a variational RNN for uncertainty estimation

Implement temporal convolutional networks

Train a model for multivariate time series classification
```

## Self-Supervised Learning

```
Implement SimCLR for image representation learning

Train a BYOL model without negative samples

Build a masked autoencoder (MAE) for vision

Implement contrastive predictive coding (CPC)

Train a CLIP-style vision-language model

Build a self-supervised model for audio

Implement DINO for self-distillation

Train a VICReg model with variance regularization

Build a Barlow Twins model

Implement SwAV with online clustering
```

## Model Compression

```
Prune a neural network to 90% sparsity

Quantize a model to 8-bit integers

Implement knowledge distillation

Build a lottery ticket subnetwork

Apply low-rank factorization to weight matrices

Implement dynamic neural networks with early exit

Quantization-aware training for deployment

Structured pruning by removing entire filters

Neural architecture search for efficient models

Implement weight sharing for compression
```

## Interpretability

```
Generate saliency maps for image classification

Implement integrated gradients attribution

Build attention visualization for transformers

Compute SHAP values for a tabular model

Generate counterfactual explanations

Implement concept activation vectors (CAVs)

Build a prototype-based interpretable model

Analyze neuron activations across layers

Implement layer-wise relevance propagation

Generate natural language explanations
```
