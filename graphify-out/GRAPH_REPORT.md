# Graph Report - .  (2026-04-19)

## Corpus Check
- 14 files · ~18,824 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 173 nodes · 279 edges · 25 communities detected
- Extraction: 68% EXTRACTED · 32% INFERRED · 0% AMBIGUOUS · INFERRED: 89 edges (avg confidence: 0.77)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Instrumentation & Config|Instrumentation & Config]]
- [[_COMMUNITY_Benchmarks & Extensions|Benchmarks & Extensions]]
- [[_COMMUNITY_Data Loading|Data Loading]]
- [[_COMMUNITY_Baseline Agent System|Baseline Agent System]]
- [[_COMMUNITY_LatentMAS Core|LatentMAS Core]]
- [[_COMMUNITY_Mechanistic Interpretability|Mechanistic Interpretability]]
- [[_COMMUNITY_HuggingFace Dependencies|HuggingFace Dependencies]]
- [[_COMMUNITY_Game Theory Dynamics|Game Theory Dynamics]]
- [[_COMMUNITY_Superposition & Features|Superposition & Features]]
- [[_COMMUNITY_Deception & Faithfulness|Deception & Faithfulness]]
- [[_COMMUNITY_Information Bottleneck|Information Bottleneck]]
- [[_COMMUNITY_Emergent Flocking|Emergent Flocking]]
- [[_COMMUNITY_Trojan Safety Risk|Trojan Safety Risk]]
- [[_COMMUNITY_Hebbian Plasticity|Hebbian Plasticity]]
- [[_COMMUNITY_Module Init|Module Init]]
- [[_COMMUNITY_Misc Community 15|Misc Community 15]]
- [[_COMMUNITY_Misc Community 16|Misc Community 16]]
- [[_COMMUNITY_Misc Community 17|Misc Community 17]]
- [[_COMMUNITY_Misc Community 18|Misc Community 18]]
- [[_COMMUNITY_Misc Community 19|Misc Community 19]]
- [[_COMMUNITY_Misc Community 20|Misc Community 20]]
- [[_COMMUNITY_Misc Community 21|Misc Community 21]]
- [[_COMMUNITY_Misc Community 22|Misc Community 22]]
- [[_COMMUNITY_Misc Community 23|Misc Community 23]]
- [[_COMMUNITY_Misc Community 24|Misc Community 24]]

## God Nodes (most connected - your core abstractions)
1. `LatentMAS Framework` - 24 edges
2. `ModelWrapper` - 21 edges
3. `main()` - 18 edges
4. `normalize_answer()` - 14 edges
5. `run_batch()` - 14 edges
6. `main()` - 13 edges
7. `run_and_capture()` - 11 edges
8. `_past_length()` - 7 edges
9. `InstrumentedLatentMAS` - 7 edges
10. `LatentMASMethod` - 7 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `ModelWrapper`  [INFERRED]
  run.py → models.py
- `main()` --calls--> `BaselineMethod`  [INFERRED]
  run.py → methods/baseline.py
- `main()` --calls--> `TextMASMethod`  [INFERRED]
  run.py → methods/text_mas.py
- `main()` --calls--> `LatentMASMethod`  [INFERRED]
  run.py → methods/latent_mas.py
- `main()` --calls--> `load_mbppplus()`  [INFERRED]
  run.py → data.py

## Hyperedges (group relationships)
- **Core LatentMAS Software Stack** — run_py, models_py, latent_mas_py, text_mas_py, baseline_py, prompts_py, data_py, utils_py [EXTRACTED 1.00]
- **Hugging Face Dependency Group** — dep_transformers, dep_accelerate, dep_datasets [EXTRACTED 1.00]
- **Safety-Track Interpretability Experiments** — exp4_hidden_deception, exp8_faithfulness_gap, exp12_trojan_attacks, exp13_robustness [EXTRACTED 1.00]
- **Mechanistic Interpretability Experiments** — exp1_latent_editor, exp7_layer_contributions, exp11_induction_heads [EXTRACTED 1.00]
- **Physics and Biology Analogy Experiments** — exp15_flocking, exp16_entanglement, exp17_evolutionary_selection, exp18_symbiotic_parasitic, exp21_hebbian, exp22_predator_prey, exp25_attractor_entropy, exp27_resonance [EXTRACTED 1.00]
- **LatentMAS Evaluation Benchmark Suite** — dataset_gsm8k, dataset_aime, dataset_gpqa, dataset_arc, dataset_mbpp, dataset_humaneval, dataset_medqa [EXTRACTED 1.00]
- **Community-Driven LatentMAS Extensions** — science_latentmas, knn_latentmas, hybrid_latentmas, awareness_network, latentmas_slora, avp_protocol [EXTRACTED 1.00]

## Communities

### Community 0 - "Instrumentation & Config"
Cohesion: 0.15
Nodes (18): CaptureConfig, _decode_latent_topk(), dir_size_bytes(), fmt_bytes(), InstrumentedLatentMAS, kv_to_legacy(), load_task(), main() (+10 more)

### Community 1 - "Benchmarks & Extensions"
Cohesion: 0.08
Nodes (24): AVP Agent Vector Protocol, Awareness Network (Decentralized AI Market), AIME24/25 Dataset, ARC-Easy/Challenge Dataset, GPQA Dataset, GSM8K Dataset, HumanEval+ Dataset, MBPP+ Dataset (+16 more)

### Community 2 - "Data Loading"
Cohesion: 0.18
Nodes (19): baseline.py (Single-Agent Baseline), load_aime2024(), load_aime2025(), load_arc_challenge(), load_arc_easy(), load_gpqa_diamond(), load_gsm8k(), load_humanevalplus() (+11 more)

### Community 3 - "Baseline Agent System"
Cohesion: 0.13
Nodes (10): BaselineMethod, Agent, default_agents(), score_prediction(), build_agent_messages_hierarchical_text_mas(), build_agent_messages_sequential_text_mas(), build_agent_messages_single_agent(), TextMASMethod (+2 more)

### Community 4 - "LatentMAS Core"
Cohesion: 0.19
Nodes (12): LatentMASMethod, run_batch(), _slice_tensor(), _ensure_pad_token(), generate_latent_batch(), generate_latent_batch_hidden_state(), generate_text_batch(), _past_length() (+4 more)

### Community 5 - "Mechanistic Interpretability"
Cohesion: 0.12
Nodes (19): Alignment Matrix W_a (Ridge Regression), Induction Heads (Cross-Agent), Exp 11: Cross-Agent Induction Heads and Collaboration Circuits, Exp 1: The Latent Editor — Mechanistic Role of W_a, Exp 25: Attractor Theory of Consensus and Thermodynamic Entropy, Exp 26: Manifold Alignment and Universal Conceptual Geometry, Exp 2: Emergent Private Latent Vocabulary, Exp 3: Cross-Agent Latent Mind-Reading (+11 more)

### Community 6 - "HuggingFace Dependencies"
Cohesion: 0.29
Nodes (6): accelerate (Hugging Face), datasets (Hugging Face), numpy, torch (PyTorch), tqdm, transformers (Hugging Face)

### Community 7 - "Game Theory Dynamics"
Cohesion: 0.5
Nodes (5): Exp 18: Symbiotic vs. Parasitic Latent Agent Interactions, Exp 22: Predator-Prey Dynamics in Hierarchical Agent Competition, Exp 23: Bargaining and Power Asymmetries in Latent Space, Shapley Values for Agent Contribution, Transfer Entropy / Directed Mutual Information

### Community 8 - "Superposition & Features"
Cohesion: 0.67
Nodes (4): Superposition in Neural Networks (Elhage et al.), Exp 19: Latent Superposition of Competing Reasoning Paths, Exp 20: Monosemanticity Collapse in Shared Working Memory, Sparse Autoencoders (SAE) for Feature Decomposition

### Community 9 - "Deception & Faithfulness"
Cohesion: 0.67
Nodes (3): Chain-of-Thought Reasoning (Baseline), Exp 4: Hidden Deception and Steganographic Collusion, Exp 8: Faithfulness Gap Between Latent and Textual Reasoning

### Community 10 - "Information Bottleneck"
Cohesion: 0.67
Nodes (3): Information Bottleneck Principle, Exp 29: Latent Compression as Regularizer — The Bottleneck Effect, Rationale: Narrow Latent Bottleneck as Regularizer

### Community 11 - "Emergent Flocking"
Cohesion: 1.0
Nodes (2): Boids Flocking Rules (Reynolds 1987), Exp 15: Emergent Flocking and Herding in Latent Thought Space

### Community 12 - "Trojan Safety Risk"
Cohesion: 1.0
Nodes (2): Exp 12: Latent Trojan Attacks — Poisoning the Working Memory, Rationale: Latent Channel Creates Invisible Safety Risk

### Community 13 - "Hebbian Plasticity"
Cohesion: 1.0
Nodes (2): Hebbian Learning (Neuroscience), Exp 21: Hebbian Strengthening and Latent Neural Plasticity

### Community 14 - "Module Init"
Cohesion: 1.0
Nodes (0): 

### Community 15 - "Misc Community 15"
Cohesion: 1.0
Nodes (1): Exp 5: Quantifying Information Compression

### Community 16 - "Misc Community 16"
Cohesion: 1.0
Nodes (1): Exp 6: Information Propagation and Decay Over Rounds

### Community 17 - "Misc Community 17"
Cohesion: 1.0
Nodes (1): Exp 10: Emergent Role Specialization

### Community 18 - "Misc Community 18"
Cohesion: 1.0
Nodes (1): Exp 13: Robustness to Targeted Interventions

### Community 19 - "Misc Community 19"
Cohesion: 1.0
Nodes (1): Exp 14: Heterogeneous Expert Agent Fusion

### Community 20 - "Misc Community 20"
Cohesion: 1.0
Nodes (1): Exp 16: Latent Entanglement — Quantum-Like Correlations

### Community 21 - "Misc Community 21"
Cohesion: 1.0
Nodes (1): Exp 17: Evolutionary Selection of Latent Communication Patterns

### Community 22 - "Misc Community 22"
Cohesion: 1.0
Nodes (1): Exp 24: Latent Catastrophe Precursors — Early Warning Signals

### Community 23 - "Misc Community 23"
Cohesion: 1.0
Nodes (1): Exp 27: Resonance and Synchronization in Latent Oscillations

### Community 24 - "Misc Community 24"
Cohesion: 1.0
Nodes (1): Exp 28: Latent Niche Construction — Emergent Theory of Mind

## Knowledge Gaps
- **61 isolated node(s):** `LatentMAS arXiv Paper (2511.20639)`, `transformers (Hugging Face)`, `torch (PyTorch)`, `numpy`, `tqdm` (+56 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Emergent Flocking`** (2 nodes): `Boids Flocking Rules (Reynolds 1987)`, `Exp 15: Emergent Flocking and Herding in Latent Thought Space`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Trojan Safety Risk`** (2 nodes): `Exp 12: Latent Trojan Attacks — Poisoning the Working Memory`, `Rationale: Latent Channel Creates Invisible Safety Risk`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Hebbian Plasticity`** (2 nodes): `Hebbian Learning (Neuroscience)`, `Exp 21: Hebbian Strengthening and Latent Neural Plasticity`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Misc Community 15`** (1 nodes): `Exp 5: Quantifying Information Compression`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Misc Community 16`** (1 nodes): `Exp 6: Information Propagation and Decay Over Rounds`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Misc Community 17`** (1 nodes): `Exp 10: Emergent Role Specialization`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Misc Community 18`** (1 nodes): `Exp 13: Robustness to Targeted Interventions`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Misc Community 19`** (1 nodes): `Exp 14: Heterogeneous Expert Agent Fusion`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Misc Community 20`** (1 nodes): `Exp 16: Latent Entanglement — Quantum-Like Correlations`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Misc Community 21`** (1 nodes): `Exp 17: Evolutionary Selection of Latent Communication Patterns`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Misc Community 22`** (1 nodes): `Exp 24: Latent Catastrophe Precursors — Early Warning Signals`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Misc Community 23`** (1 nodes): `Exp 27: Resonance and Synchronization in Latent Oscillations`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Misc Community 24`** (1 nodes): `Exp 28: Latent Niche Construction — Emergent Theory of Mind`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ModelWrapper` connect `Instrumentation & Config` to `Data Loading`, `Baseline Agent System`, `LatentMAS Core`?**
  _High betweenness centrality (0.091) - this node is a cross-community bridge._
- **Why does `main()` connect `Data Loading` to `Instrumentation & Config`, `Baseline Agent System`, `LatentMAS Core`?**
  _High betweenness centrality (0.057) - this node is a cross-community bridge._
- **Why does `LatentMAS Framework` connect `Benchmarks & Extensions` to `Mechanistic Interpretability`?**
  _High betweenness centrality (0.048) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `ModelWrapper` (e.g. with `CaptureConfig` and `InstrumentedLatentMAS`) actually correct?**
  _`ModelWrapper` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `main()` (e.g. with `set_seed()` and `auto_device()`) actually correct?**
  _`main()` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `normalize_answer()` (e.g. with `score_prediction()` and `load_gsm8k()`) actually correct?**
  _`normalize_answer()` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `run_batch()` (e.g. with `process_batch()` and `build_agent_message_sequential_latent_mas()`) actually correct?**
  _`run_batch()` has 11 INFERRED edges - model-reasoned connections that need verification._