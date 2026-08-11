# Primary source register

Publication labels are adjacent to use. A preprint or lab report can be useful
without being treated as peer-reviewed consensus.

| Source | Status used here | Modules | Boundary taught |
| --- | --- | --- | --- |
| [Doshi-Velez & Kim, rigorous interpretability](https://arxiv.org/abs/1702.08608) | position paper / preprint | 00 | Evaluation must be task- and audience-specific. |
| [Lipton, Mythos of Model Interpretability](https://arxiv.org/abs/1606.03490) | workshop/preprint lineage | 00 | “Interpretability” names distinct goals. |
| [Searchless Chess](https://arxiv.org/abs/2402.04494), [official code](https://github.com/google-deepmind/searchless_chess) | NeurIPS 2024, peer-reviewed; official repository | 01–02 | Strength without external search does not identify an internal algorithm. |
| [McGrath et al., AlphaZero chess knowledge](https://pmc.ncbi.nlm.nih.gov/articles/PMC9704706/) | peer-reviewed | 01, 04 | Probe trajectories show decodability, not automatically use. |
| [Kornblith et al., representation similarity](https://proceedings.mlr.press/v97/kornblith19a.html) | ICML 2019, peer-reviewed | 03 | Metrics encode different invariances. |
| [Hewitt & Liang, probe controls](https://aclanthology.org/D19-1275/) | EMNLP-IJCNLP 2019, peer-reviewed | 04 | Capacity and memorization need controls. |
| [Voita & Titov, MDL probing](https://aclanthology.org/2020.emnlp-main.14/) | EMNLP 2020, peer-reviewed | 04 | Sample efficiency matters beyond final accuracy. |
| [Karvonen, Chess-GPT world model](https://arxiv.org/abs/2403.15498) | COLM 2024, peer-reviewed | 04–05 | Decodability and steering need bounded claims. |
| [Sundararajan et al., Integrated Gradients](https://proceedings.mlr.press/v70/sundararajan17a.html) | ICML 2017, peer-reviewed | 05 | Baseline and path define the attribution estimand; completeness is a diagnostic, not causality. |
| [Smilkov et al., SmoothGrad](https://arxiv.org/abs/1706.03825) | 2017 preprint | 05 | Noise-averaged sensitivity depends on the perturbation distribution. |
| [Adebayo et al., Sanity Checks for Saliency Maps](https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html) | NeurIPS 2018, peer-reviewed | 05 | Model- and label-randomization tests are necessary attribution controls. |
| [Puri et al., SARFA](https://nikaashpuri.github.io/sarfa-saliency/) | ICLR 2020, peer-reviewed | 05 | Perturbation attribution should preserve action specificity and relevance. |
| [Jain & Wallace, Attention is not Explanation](https://aclanthology.org/N19-1357/) | NAACL 2019, peer-reviewed | 05 | Attention weights alone are not a mechanism. |
| [Wiegreffe & Pinter, Attention is not not Explanation](https://aclanthology.org/D19-1002/) | EMNLP 2019, peer-reviewed | 05 | Specify the explanatory question and counterfactual. |
| [Meng et al., ROME / causal tracing](https://proceedings.neurips.cc/paper_files/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html) | NeurIPS 2022, peer-reviewed | 05 | Patching localizes a scoped mediator. |
| [Linear Pin Representations](https://openreview.net/forum?id=sbt3pBy9Rx) | ICML 2026 Mechanistic Interpretability Workshop virtual poster | 05 | Strong probes/random controls can overstate behavioral causation. |
| [Jenner et al., Learned Look-Ahead](https://proceedings.neurips.cc/paper_files/paper/2024/hash/37d9f19150fce07bced2a81fc87d47a6-Abstract-Conference.html) | NeurIPS 2024, peer-reviewed | 06 | Probe, intervention, and scope are separate evidence lines. |
| [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) | lab report | 07 | Sparse features are not proof of monosemanticity. |
| [Transcoders](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2b8f4db0464cc5b6e9d5e6bea4b9f308-Abstract-Conference.html) | NeurIPS 2024, peer-reviewed | 07–08 | A transcoder replaces a computation boundary. |
| [Sparse Crosscoders](https://transformer-circuits.pub/2024/crosscoders/index.html) | lab report | 08 | Cross-model features remain representational until intervention-linked. |
| [Tracing the Thought of a Grandmaster-level Chess-Playing Transformer](https://arxiv.org/abs/2604.10158) | 2026 preprint | 08–09 | Sparse chess tooling must be source- and ABI-matched. |
| [Geiger et al., Causal Abstraction](https://arxiv.org/abs/2301.04709) | JMLR 2025, peer-reviewed | 09 | Interchange tests a mapping to an abstract causal model. |
| [ACDC](https://proceedings.neurips.cc/paper_files/paper/2023/hash/34e1dbe95d34d7ebaf99b9bcaeb5b2be-Abstract-Conference.html) | NeurIPS 2023, peer-reviewed | 09 | Discovery inherits metric and graph assumptions. |
| [MIB](https://proceedings.mlr.press/v267/mueller25a.html) | ICML 2025, peer-reviewed | 09–10 | Sparse features need not beat neurons. |
| [marimo documentation](https://docs.marimo.io/) | official documentation | all | Reactive execution helps reproducibility; PyTorch is not assumed to run in browser WASM. |

Each notebook labels sources locally as peer-reviewed, preprint, lab report, or
official documentation and links back to this register.
