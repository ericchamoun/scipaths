# SciPaths

Code and data release for "SCIPATHS: Forecasting Pathways to Scientific Discovery".

Scientific progress often depends on prior methods, datasets, tools, empirical findings, and conceptual advances that make a new contribution possible. This raises a central question for scientific forecasting: given a target contribution, which enabling contributions are required to realize it?

SciPaths investigates the ability of LLMs to generate and ground these contribution pathways, and provides a benchmark, public dataset, evaluation code, and silver annotation pipeline for studying this problem.

Paper: https://arxiv.org/pdf/2605.14600

## Contents

| Directory | Purpose |
| --- | --- |
| `data/` | Public claim-level training, development, and test splits. |
| `benchmark_eval/` | Evaluation code for enabling contribution generation, grounding, and end-to-end runs. |
| `silver_annotation_pipeline/` | Automatic annotation pipeline for producing silver target-contribution pathway annotations at scale from arXiv papers. |

See each directory README for setup, commands, and output formats.

## Overview

### Data

`data/` contains the public claim-level release:

- `training.json`: silver annotations for training and analysis.
- `dev.json`: gold development annotations.
- `test.json`: blind test inputs with target contributions only.

Each labeled example contains a target contribution, enabling contributions, primary groundings, and additional groundings.

### Benchmark Evaluation

`benchmark_eval/` contains scripts for:

- generating enabling contributions from a target contribution;
- judging generated enabling contributions against gold or silver annotations;
- grounding target contributions or enabling contributions in prior papers;
- running generation and grounding end-to-end.

The benchmark reports enabling-contribution generation metrics and grounding retrieval metrics.

### Silver Annotation Pipeline

`silver_annotation_pipeline/` contains the automatic annotation pipeline used to scale SciPaths-style annotations to new arXiv papers.

It fetches a paper, finds downstream citation contexts, verifies USES/EXTENDS relationships, derives reusable target contributions from downstream impact, and decomposes those target contributions into enabling contributions and grounded studies. Outputs are saved locally under `silver_annotation_pipeline/runs/`.

## Demo

🤗 Try SciPaths on your own arXiv paper and see how later work builds on it:

https://huggingface.co/spaces/EricCham8/Scipaths

The demo follows the downstream impact of a paper, groups citing papers by how they use or extend it, identifies reusable target contributions, and decomposes each one into enabling contributions with grounded prior studies.

## Citation

If you find this useful, please cite our paper as:

```bibtex
@misc{chamoun2026scipathsforecastingpathwaysscientific,
      title={SciPaths: Forecasting Pathways to Scientific Discovery}, 
      author={Eric Chamoun and Yizhou Chi and Yulong Chen and Rui Cao and Zifeng Ding and Michalis Korakakis and Andreas Vlachos},
      year={2026},
      eprint={2605.14600},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.14600}, 
}
```

Paper URL: https://arxiv.org/abs/2605.14600

For questions or comments, contact ec806@cam.ac.uk.

## License

The code in this repository is released under the MIT License. See `LICENSE` for details.

Data files and paper-derived annotation content may remain subject to the licenses and terms of the underlying scholarly sources.
