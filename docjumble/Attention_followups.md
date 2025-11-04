# Attentions 
* read follow ups

## 2D Rotary Positional Embeddings (2D RoPE)
1. Original 1D RoPE Paper  Su, J., et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding". arXiv:2104.09864
    * core mathematical concept of rotating query and key vectors to encode relative position.
2. Vision (2D RoPE) Heo, B., et al. (2024). "Rotary Position Embedding for Vision Transformer". (Appeared in ECCV 2024). arXiv:2403.09605
    * formally analyzes and adapts 1D RoPE to 2D vision. 
    * "Axial" RoPE applies 1D RoPE along the x-axis and y-axis independently to give ViTs strong 2D spatial awareness and impressive extrapolation performance (e.g., training on 256x256 images and inferring at 1024x1024).
3. 3D Rope. Learning the RoPEs: Better 2D and 3D Position Encodings with STRING". [arXiv:2502.02562]
    * learns position encodings for 2D and 3D data, robotics and 3D object detection.

## On Quaternions and RoPE
1. Li, Z., et al. (2024). "QEAN: Quaternion-Enhanced Attention Network for Visual Dance Generation". arXiv:2403.11626Contribution
    * "Quaternion Rotary Attention (QRA)" module to model complex spatio-temporal data (a dancing human body). . 
    * v * a + bi + cj + dk -> v_rot 
    * can encode multiple dimensions of positional information (e.g., space, time, or frequency) into a single, efficient rotational embedding
2. JCAI 2024. "QFormer: An Efficient Quaternion Transformer for Image Denoising".
    * "Quaternion Transformer" that represents color pixels as quaternions.
    * self-attention (R, G, B)  preserving their internal correlations.
3. Video / 3D (for Spatio-temporal data) Li, Z., et al. (2024). "QEAN: Quaternion-Enhanced Attention Network for Visual Dance Generation". arXiv:2403.11626
    * "Quaternion Rotary Attention" (QRA) module. Fuses quaternions 3D joint rotations of a human body , and RoPE-like to encode their position in time.
## DDIM Inversion
"DDIM Inversion" is a technique made possible by the original DDIM paper, which introduced a deterministic sampling process.
1. The Foundational Paper (DDIM): Song, J., Meng, C., & Ermon, S. (2020). "Denoising Diffusion Implicit Models". arXiv:2010.02502
    * DDPM (Denoising Diffusion Probabilistic Models) paper described a stochastic (random) sampling process.
    * DDIM reformulated this as a deterministic process.
    * Why it's key for inversion: Because the process is deterministic, if you know the final image $x_0$ and the noise $\epsilon_t$ predicted at a step $t$, you can algebraically solve for the latent $x_t$ from the previous step. DDIM Inversion is simply running this deterministic process in reverse to "re-noise" a clean image back to its initial pure-noise latent, $z_T$.
2. P Zhang, Z., et al. (2025). "EasyInv: Toward Fast and Better DDIM Inversion". (Appeared in ICML 2025).
    * SOTA method, Speed and efficienty addressing the performance limitations and is compared against other recent methods like "Fixed-Point Iteration" (FPI).
3. ECCV 2024. "bi-directional integration approximation (BDIA)".
    * a novel technique to achieve mathematically exact diffusion inversion with very low computational overhead. This is critical for video, where errors accumulate over frames.
4. NeurIPS 2024. "Exploring Fixed Point in Image Editing".
    * theoretical proof (using the Banach fixed-point theorem) for why iterative inversion methods converge. 
    * mathematical foundation for SOTA methods like FPI and EasyInv
* huggigface.diffusers DDPMScheduler and DDIMScheduler both support invert_latents

## RLAIF (Reinforcement Learning from AI Feedback)
Evolved from RLHF (RL from Human Feedback). replace the expensive human labeler with a powerful "judge" AI.
1. The Foundational Concept Paper:  Bai, Y., et al. (2022). "Constitutional AI: Harmlessness from AI Feedback". arXiv:2212.08073
    * Contribution: This paper from Anthropic introduced the core workflow. An AI model generates responses, and a separate, powerful AI provides feedback (e.g., "Which response is more harmless?"). This AI-generated feedback is then used to train a Reward Model, which in turn trains the original model via RL.
2. Lee, H., et al. (2023). "RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback". arXiv:2309.00267
    * "RLAIF" term.
    *  off-the-shelf LLM as a preference labeler can match, and in some cases exceed, the performance of models trained with human feedback (RLHF), at a fraction of the cost.

## VQA (Visual Question Answering)
1. The Seminal "VQA" Paper Antol, S., et al. (2015). "VQA: Visual Question Answering". (Appeared in ICCV 2015). arXiv:1505.00468
    * established the VQA task as a benchmark.
    * Introduced the first large-scale, open-ended VQA dataset and provided the initial baseline models
2. VQA (SOTA Baseline) Li, J., et al. (2023). "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models". arXiv:2301.12597
    * BLIP-2 and its successors (like LLaVA) are the dominant architecture. 
    * bridge a frozen image encoder (ViT) to a frozen LLM (like FLAN-T5) with a trainable "Q-Former." : foundation of most modern LMMs.
3. Video VQA: "SkyReels-V2: An Infinite-length Film Generative Model".
    * Generative + Multimodal Large Language Model (MLLM) with its diffusion framework.
    * MLLM is used for deep video understanding and prompt adherence;  complex commands.
4. 3D VQA:"Spann3R: Dense 3D Reconstruction from Ordered or Unordered Image Collections".  3DV 2025.
    * use a Transformer to regress a "global pointmap, The next ste: connect 3D "world models" to LLMs, like BLIP for 2D images.