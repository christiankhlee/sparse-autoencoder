# Sparse Feature Tracker: Probing Feature Stability Across Paraphrased Factual-Recall Prompts

**Author:** Christian
**Date:** March 2026
**Model:** GPT-2 small
**SAE:** gpt2-small-res-jb, layer 8 residual stream

---

## Abstract

We investigate whether sparse latent features learned by a Sparse Autoencoder (SAE) trained on GPT-2 small's layer-8 residual stream are stable across semantically equivalent, paraphrased prompts for factual recall tasks. Using 18 prompt families covering capital cities, famous inventors, and world geography facts, we extract the top-20 active SAE features for each prompt and measure pairwise Jaccard overlap within families. We find that [INSERT FINDING: mean stability score], indicating that [INSERT CONCLUSION: sparse features are/are not highly stable]. We additionally compare feature activation patterns between prompts where the model predicts the correct answer versus incorrect ones, identifying a set of differentiating features that may encode factual confidence or topic identity. This work contributes to the growing toolkit of mechanistic interpretability methods and provides a reusable pipeline for sparse feature analysis on pretrained language models.

---

## 1. Introduction and Motivation

Large language models (LLMs) encode vast amounts of factual knowledge in their weights, yet the internal representations underlying this knowledge remain poorly understood. Mechanistic interpretability aims to reverse-engineer the computational structures that enable factual recall—identifying which neurons, circuits, or features implement specific behaviors.

Sparse Autoencoders (SAEs) offer a promising lens. By training a sparse bottleneck network to reconstruct a model's activations, SAEs learn a dictionary of "features"—directions in activation space that (a) tend to activate on semantically coherent inputs and (b) are approximately independent of one another (sparse). Recent work from Bricken et al. (2023) and Cunningham et al. (2023) has demonstrated that many SAE features are human-interpretable, monosemantic, and causally involved in model behavior.

This project asks a specific empirical question: **are the same SAE features active when GPT-2 small processes semantically equivalent but lexically different prompts for the same factual query?** If sparse features are truly encoding factual knowledge (rather than surface-level syntax), we would expect high feature-set overlap across paraphrases. Conversely, low overlap would suggest that factual recall in GPT-2 is more syntactically-driven, or that the SAE features at layer 8 encode earlier, more syntactic phenomena.

A secondary question: **do different SAE features activate when the model answers correctly versus incorrectly?** If so, these differentiating features may be interpretability handles for factual confidence, topic identity, or error modes.

---

## 2. Methods

### 2.1 Model

We use **GPT-2 small** (117M parameters, 12 transformer layers, d_model = 768), loaded from Hugging Face via the `transformers` library. GPT-2 small is an ideal testbed for interpretability research: small enough to run on CPU/laptop hardware, well-studied, and covered by the sae_lens pretrained SAE collection.

### 2.2 Sparse Autoencoder

We load the **gpt2-small-res-jb** SAE release from `sae_lens`, specifically the SAE trained on the **layer-8 residual stream** (`blocks.8.hook_resid_post`). This SAE was trained by Joseph Bloom (JB) using the methodology described in Cunningham et al. (2023). The SAE maps from d_model = 768 to a substantially wider feature space (typically 24,576 features), enforcing sparsity through an L1 penalty during training.

For each prompt, we:
1. Run a forward pass through GPT-2 small with a forward hook registered on `transformer.h[8]`.
2. Extract the residual-stream activation for the **final token** (the token immediately before the model's predicted next token).
3. Encode this 768-dimensional vector through the SAE using `sae.encode()`, yielding a sparse feature activation vector.
4. Identify the **top-20 active features** (by activation magnitude).

### 2.3 Prompt Families

We curated 18 **prompt families** covering three categories:

| Category | Examples | Families |
|----------|----------|----------|
| Capital cities | France→Paris, Japan→Tokyo, Germany→Berlin, Italy→Rome, Spain→Madrid, Australia→Canberra, Brazil→Brasília, Canada→Ottawa | 8 |
| Famous inventors | telephone→Bell, lightbulb→Edison, gravity→Newton, relativity→Einstein, evolution→Darwin | 5 |
| World facts | largest country→Russia, longest river→Nile, highest mountain→Everest | 3 |

Each family contains:
- **4–5 paraphrased prompts**: lexically varied but semantically identical, all expecting the same correct completion.
- **2 distractor prompts**: related but subtly different queries that should elicit a different (or "incorrect" relative to the family) answer.

Paraphrase variation includes: different syntactic structures ("The capital of X is" vs "In X, the capital is"), question-answer frames ("What is the capital of X? The answer is"), and possessive constructions ("X's capital city is").

### 2.4 Metrics

**Jaccard Similarity** measures feature-set overlap between two prompts:

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

where A and B are the top-20 feature index sets for two prompts. Values range from 0 (no shared features) to 1 (identical feature sets).

**Family Stability Score**: the mean pairwise Jaccard similarity across all prompt pairs within a family.

**Differentiating Feature Score**: the difference in frequency (across prompts) between correct-prediction contexts and incorrect-prediction contexts for each feature.

---

## 3. Results

### 3.1 Feature Stability Across Paraphrases

[INSERT TABLE 1: Consistency results table — family_id, n_prompts, mean_overlap, std_overlap, min_overlap, max_overlap]

[INSERT FIGURE 1: Bar chart of mean Jaccard overlap per family (fig1_overlap_by_family.png)]

Overall, we observe [INSERT FINDING: describe the range of stability scores observed]. Families with [INSERT PATTERN, e.g., shorter prompts / specific topic types] tend to show higher stability, suggesting [INSERT INTERPRETATION].

The mean stability score across all families is **[INSERT VALUE]** (range: [INSERT MIN]–[INSERT MAX]). This is [INSERT COMPARISON: above/below] chance overlap (expected Jaccard for two random size-20 subsets of 24,576 features ≈ 0.0016), indicating that the model does use a partially consistent feature representation across paraphrases.

### 3.2 Feature Activation Heatmap

[INSERT FIGURE 2: Feature activation heatmap (fig2_feature_heatmap.png)]

The heatmap reveals [INSERT FINDING: describe visible structure, e.g., clusters of prompts sharing active features, features that are family-specific vs universal].

### 3.3 Correct vs Incorrect Prediction Features

[INSERT TABLE 2: Top 10 differentiating features, with correct_count, incorrect_count, difference]

[INSERT FIGURE 3: Correct vs incorrect feature chart (fig3_correct_vs_incorrect.png)]

The model predicted the correct next token on **[INSERT VALUE]%** of paraphrase prompts. Features most associated with correct predictions (positive difference) include [INSERT FEATURE INDICES AND ANY INTERPRETATION]. Features most associated with incorrect predictions include [INSERT FEATURE INDICES].

---

## 4. Discussion

The results suggest that sparse features in GPT-2 small's layer-8 residual stream exhibit [INSERT LEVEL: partial/high/low] stability across paraphrased factual recall prompts. Several interpretations are possible:

1. **Stable features encode topic identity.** Features that appear consistently across paraphrases of "capital of France" may directly encode the concept of France or the concept of European capital cities, rather than surface-level syntax.

2. **Unstable features encode syntactic framing.** The variation in features across paraphrases may reflect how the model tracks different sentence structures, question formats, or discourse contexts—all of which change across paraphrases even when the semantic content is constant.

3. **Layer 8 may be a transition zone.** Earlier layers in GPT-2 are known to process more syntactic features; later layers encode more semantic content. Layer 8 may exhibit a mixture of both, explaining intermediate stability scores.

---

## 5. Limitations

- **Top-1 token evaluation**: We evaluate correctness only on the top-1 predicted next token, which may undersell actual performance (the correct answer may be in the top-5).
- **Single layer**: We only analyze layer 8. A richer picture would compare stability across all 12 layers.
- **SAE coverage**: SAEs do not perfectly reconstruct all model behavior; some features may be in the unexplained residual.
- **Small model**: GPT-2 small has limited factual knowledge capacity compared to modern large models; results may not generalize.
- **Prompt design**: Our paraphrases were hand-crafted; systematic paraphrase generation (e.g., using a paraphrase model) would improve rigor.
- **No causal interventions**: This study is purely observational. We do not patch or ablate features to confirm their causal role.

---

## 6. Future Work

1. **Multi-layer analysis**: Track which layer first shows stable feature activation for each family; map the "where does factual recall happen?" question.
2. **Feature causal validation**: Use activation patching to verify that the identified features causally contribute to correct predictions.
3. **Larger SAE feature interpretation**: Decode the top differentiating features using automated interpretability tools (e.g., neuron2graph, SAE feature dashboards).
4. **Systematic paraphrase generation**: Generate paraphrases using T5 or GPT-4 for higher coverage and diversity.
5. **Scaling to larger models**: Replicate on GPT-2 medium, GPT-2 large, or Pythia models with SAE coverage from sae_lens.
6. **Cross-family analysis**: Study whether features that encode "capital city" knowledge generalize across all capital-city families.

---

## References

- Bricken, T., Templeton, A., Batson, J., et al. (2023). *Towards Monosemanticity: Decomposing Language Models with Dictionary Learning*. Anthropic.
- Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2023). *Sparse Autoencoders Find Highly Interpretable Features in Language Models*. ICLR 2024.
- Bloom, J. (2024). *sae-lens: A Library for Training and Analysing Sparse Autoencoders*. GitHub.
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). *Language Models are Unsupervised Multitask Learners*. OpenAI.
