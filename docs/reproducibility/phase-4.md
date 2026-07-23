# Phase 4 — Historical environment recovery

## Status

A minimal CPU execution path has been recovered for the archived distance-based implementation. This is a compatibility reconstruction, not proof of the exact original machine environment or reproduction of the published results.

## Verified compatibility environment

The GitHub Actions smoke test uses:

- Ubuntu 22.04 CPU runner
- Python 3.8
- PyTorch 1.8.1 CPU
- torchvision 0.9.1 CPU
- FastAI 2.3.1
- FastCore 1.3.20
- spaCy 2.3.1
- NumPy 1.20.3
- SciPy 1.6.3

The full reconstructed dependency set is stored in `environment/legacy-requirements.txt`. The original repository did not record a complete environment lockfile, so these versions were selected to match the 2021-era notebooks and APIs and then verified through GitHub Actions.

## Verified execution path

The command below has been verified on CPU:

```bash
python scripts/smoke_test.py
```

It performs the following operations without modifying the archived implementation:

1. constructs a reduced one-layer ARViT2D instance;
2. creates the historical 2D penalty matrix;
3. completes a forward pass;
4. verifies logits, raw attention, reduced attention, and penalty-matrix dimensions;
5. evaluates the historical `ARViT2D_Loss`;
6. performs backward propagation and one SGD optimizer step.

The following command has also been verified:

```bash
python scripts/smoke_test.py --imagenette
```

It downloads Imagenette-160 and repeats the same one-batch training smoke test using real images.

## Installation

```bash
python -m pip install pip==23.3.2 setuptools==68.2.2 wheel==0.41.3
python -m pip install -r environment/legacy-requirements.txt
```

## Historical launch scripts

The archived launch scripts are not used by the smoke test. They assume:

- CUDA and NCCL distributed execution;
- a `--local_rank` argument;
- FastAI distributed training;
- FP16 execution;
- machine-specific paths under `Path.home() / 'Luiz/...'`;
- full dataset and output directories that are not included in the repository.

Those assumptions are preserved as historical evidence and will be addressed only in later modernization phases.

## Checkpoint status

Run:

```bash
python scripts/check_checkpoints.py
```

The files under `pretrained_models/` are historical Git LFS pointer entries in the migrated repository. The corresponding binary objects were not migrated, so checkpoint loading cannot currently be verified. This remains an explicit Phase 4 blocker and will be revisited during checkpoint and artifact recovery.

## Scope limitations

This phase does not establish:

- exact reproduction of the original CUDA environment;
- full pretraining or downstream fine-tuning;
- published accuracy reproduction;
- checkpoint equivalence;
- correction of the paper/code discrepancies identified in Phase 2;
- support for image or patch dimensions outside the historical 256 × 256 configuration.

No archived model, regularizer, loss, or launch-script behavior was changed in this phase.
