# Technical Deep Dive: Text-to-Music Generation Pipeline

## 1. Model Architectures & Training Dynamics

This project implements a *Language Modeling* approach to audio generation. Instead of predicting continuous waveforms (which is high-dimensional and unstable), we treat audio as a sequence of discrete tokens derived from a neural audio codec (EnCodec).

### 1.1 The Core Paradigm: Audio as discrete Tokens
We utilize *EnCodec*, a convolutional autoencoder with Residual Vector Quantization (RVQ).
- *Encoder:* Transforms raw audio $x \in \mathbb{R}^{T_{samples}}$ into a latent representation $z$.
- *Quantizer:* Maps $z$ to nearest neighbors in $N_q$ codebooks.
- *Representation:* The audio is represented as a matrix $S \in \{1, ..., K\}^{T_{frames} \times N_q}$, where $K=2048$ is the codebook size and $N_q=4$ is the number of codebooks (quantizers).

### 1.2 Model A: Baseline (Text-to-Music)
*Architecture:* Standard Transformer Decoder (GPT-2 style).
* *Input Processing:*
    * *Text:* Tokenized via T5-Base $\rightarrow$ Fixed Embeddings ($768d$).
    * *Audio Codes:* The discrete codes are flattened or interleaved. We use an *Interleaved Pattern*: $[Code_{1,t}, Code_{2,t}, Code_{3,t}, Code_{4,t}, Code_{1,t+1}, ...]$.
    * *Embedding Layer:* Learned embeddings map discrete code indices $[0..2048]$ to $D_{model}$ dimensions.
* *Conditioning:* Cross-Attention is not used in the simplest baseline; instead, text embeddings are prepended to the sequence (Prefix LM).
* *Forward Pass:*
    $$h_0 = \text{Concat}(\text{Emb}{text}, \text{Emb}{audio})$$
    $$h_{i} = \text{TransformerBlock}(h_{i-1})$$
    $$\text{Logits} = \text{Linear}(h_L) \in \mathbb{R}^{V_{vocab}}$$
* *Loss Function:* Cross Entropy Loss.
    $$\mathcal{L}{CE} = - \sum{t} \log P(s_t | s_{<t}, \text{text})$$
    We ignore padding tokens using ignore_index=-100.

### 1.3 Model B: Proposed (Text + Audio Prompting)
*Architecture:* Multi-Modal Transformer Decoder.
* *Inputs:*
    1.  *Text:* T5 Embeddings (Semantic intent).
    2.  *Audio Prompt:* A short clip (~3s) encoded via EnCodec to seed the style/tempo.
* *Fusion Mechanism:*
    * We employ *Cross-Attention* layers interleaved with Self-Attention.
    * Query ($Q$): Current audio generation stream.
    * Key/Value ($K, V$): Concatenation of Text and Audio Prompt embeddings.
* *Forward & Backward Propagation:*
    1.  *Forward:* The model takes the conditioning $C$ and the target sequence $S$ (shifted right). It outputs logits for the next codebook index.
    2.  *Backprop:* Gradients flow through the Transformer blocks. Note: Gradients *do not* flow into the T5 encoder or EnCodec (they are frozen).
    3.  *Optimizer:* AdamW.
        * $\beta_1=0.9, \beta_2=0.99$.
        * Weight Decay: $0.1$ (to prevent overfitting on small datasets).
        * Learning Rate: Cosine Annealing with Warmup (Linear warmup for first 10% of steps).

### 1.4 Optimizer & Hyperparameters
* *Precision:* Mixed Precision (16-mixed) via PyTorch Lightning to reduce VRAM usage on T4 GPUs.
* *Gradient Clipping:* Norm capped at $1.0$ to prevent exploding gradients in deep transformer stacks.
* *Batch Size:* Dynamic (typically 4-8 on Colab T4), requiring Gradient Accumulation to simulate effective batch sizes of 32+.

---

## 2. Machine Learning Model Selection & Justification

### 2.1 Why Discrete Audio Modeling (Transformers) vs. Diffusion?
* *Diffusion (e.g., AudioLDM):* Generates Mel-spectrograms which are then vocoded.
    * Pros: High quality, continuous.
    * Cons: Slow inference (many denoising steps), harder to condition on specific temporal structures (beats).
* *Discrete Transformers (MusicGen/AudioLM):*
    * Pros: *Fast inference* (optimized caching), explicit temporal dependency modeling (perfect for rhythm), leverages massive improvements in NLP (scaling laws).
    * Selection: We chose the Transformer approach because it allows us to frame music generation as a sequence-to-sequence translation problem, making the pipeline robust and mathematically similar to standard NLP tasks.

### 2.2 Why EnCodec?
* EnCodec is currently the SOTA neural audio codec, offering high fidelity at very low bitrates (3kbps - 6kbps). This compactness is crucial for training on limited compute (Colab), as it reduces the sequence length the Transformer must learn.

---

## 3. Model Updates & Evaluation Strategy

### 3.1 Update Strategy
* *Checkpointing:* We save the top-3 checkpoints based on *Validation Loss*.
* *EMA (Exponential Moving Average):* (Optional) We maintain an EMA of model weights during training for more stable generation results, smoothing out stochastic noise in the final epochs.

### 3.2 Evaluation Metrics (Quantitative)
1.  *Reconstruction Loss (NLL):* The negative log-likelihood on the held-out test set. Measures how well the model predicts the next code.
2.  *CLAP Score (Frechet Audio Distance - Proxy):*
    * We use the *CLAP (Contrastive Language-Audio Pretraining)* model.
    * We encode the generated audio and the input text prompt.
    * *Metric:* Cosine Similarity between these two embeddings. High similarity $\implies$ better text adherence.
3.  *FAD (Fréchet Audio Distance):*
    * Measures the distribution distance between the background "MusicCaps" embeddings and the generated audio embeddings. Lower is better (more realistic).

### 3.3 Evaluation (Qualitative)
* *WandB Audio Logging:* Every $N$ epochs, we generate samples for a fixed set of prompts:
    * "A fast techno beat with a synth lead."
    * "A relaxing piano melody with rain sounds."
    * "Heavy metal guitar riff."
* Human listening tests (informal) to judge musicality and artifacting.

---

## 4. Literature Survey: Comparison & Classification

The field of neural audio generation has evolved from signal processing to deep generative modeling.

| Model | Type | Architecture | Conditioning | Strengths | Weaknesses |
| :--- | :--- | :--- | :--- | :--- | :--- |
| *Jukebox (OpenAI)* | VQ-VAE + Sparse Transformer | Autoregressive (PixelCNN++) | Lyrics, Artist, Genre | Long-range coherence, singing. | Extremely slow inference/training. |
| *MusicLM (Google)* | Hierarchical Seq2Seq | Transformer (Semantic $\rightarrow$ Acoustic) | Text, Melody | High fidelity, strictly consistent. | Complex multi-stage cascade. |
| *AudioLM (Google)* | Language Model | GPT-style (Semantic + Acoustic tokens) | Audio continuation | High consistency, audio quality. | Unconditional or Audio-only (originally). |
| *MusicGen (Meta)* | Single-Stage LM | Transformer (Interleaved Codebooks) | Text, Melody | *Fast*, simple architecture, efficient. | Can struggle with very long context. |
| *Our Implementation* | *MusicGen-Lite* | *Single-Stage Transformer* | *Text + Audio* | *Lightweight (Colab-friendly), Multimodal.* | *Lower fidelity (small model size).* |

*Classification:*
* *Waveform Domain:* WaveNet, SampleRNN (Obsolete).
* *Spectrogram Domain:* AudioLDM, Riffusion (Diffusion-based).
* *Latent/Token Domain:* MusicGen, AudioLM, Jukebox (Transformer-based). *(Our Approach)*

---

## 5. Proposed New / Ensemble Ideas

### 5.1 Multi-Stream Conditioning (Implemented in Model B)
Most baselines condition only on text or only on audio. Our *Model B* introduces a *fusion mechanism* where the model can "style transfer" a user-uploaded audio clip (e.g., a drum beat) while following the text instruction (e.g., "make it sound like 80s synth pop").
* Mechanism: The audio prompt is encoded into tokens, embedded, and then projected into the same dimension as text embeddings. These are concatenated to form a rich "Context Memory" for the Transformer's Cross-Attention layers.

### 5.2 Future Extension: Classifier-Free Guidance (CFG) for Audio
* Borrowed from Diffusion models, we can apply CFG to Transformers.
* During inference, we compute two logits:
    1.  *Unconditional:* $P(s_t | s_{<t}, \emptyset)$
    2.  *Conditional:* $P(s_t | s_{<t}, \text{text})$
* *Formula:* $\text{Logits} = \text{Logits}{uncond} + w \cdot (\text{Logits}{cond} - \text{Logits}_{uncond})$.
* Benefit: Drastically improves adherence to the text prompt at the cost of diversity.

### 5.3 Ensemble Decoding
* Instead of training an ensemble, we propose *Reranking*:
    1.  Generate $N=4$ candidates for a single prompt.
    2.  Score each candidate using the pre-trained CLAP model (Text-Audio similarity).
    3.  Return the candidate with the highest CLAP score.
    * Justification: This is computationally cheaper than training larger models and filters out "bad" generations (hallucinations).
