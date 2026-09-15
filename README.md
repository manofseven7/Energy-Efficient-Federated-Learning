# Energy-efficient federated multi-task learning: reproducibility archive

This archive reconstructs and corrects the method in the reviewed manuscript.
It is deliberately separate from the earlier LinUCB/PCGrad/adaptive-
quantization prototype; no result from that different algorithm is used here.
The evidence is a public-data simulation study, not a reproduction of the lost
80-board power traces.

## Contents

- `src/core.py`: task-aware TopK masks, error feedback, payload accounting,
  cancellation-safe freezing statistic, and feasible scheduling.
- `src/experiment.py`: FedAvg, FedProx, Laplacian FMTL, TopK-SGD, SCAFFOLD,
  and the proposed method.
- `src/preprocess_public.py`: UCI HAR and CWRU preprocessing (plus optional
  legacy PAMAP2 support); `src/preprocess_intel.py`: chronological Intel Lab
  preprocessing.
- `src/analyze.py` and `src/analyze_extended.py`: fixed-seed aggregation,
  statistics, telemetry, ablations, and sensitivity figures.
- `tests/`: mathematical, preprocessing, and integration regression tests.
- `data/`: intentionally empty in the GitHub distribution. Public datasets are **not committed**; download them from their original providers and preprocess them locally as described below. The published `results/` are retained unchanged.
- `results/analysis/` and `results/analysis_extended/`: directly browsable aggregate tables and figures.
- `results/archive/raw_experiment_runs.tar.gz`: **all original per-run manifests and CSV logs**, losslessly bundled into one archive to avoid GitHub browser file-count limits. The archive includes `results/v2` (the corrected scheduler rerun) and the earlier audit-history runs.
- `results/archive/raw_experiment_runs.contents.txt`: exact inventory of every file stored in the raw-results archive.
- `paper/`: IEEEtran manuscript, bibliography, response letter, and figures.

The executable contract and evidentiary boundaries are documented in
`METHOD_CONTRACT.md` and `RESULTS_AUDIT.md`.

## Environment and tests

Python 3.11 or newer is recommended. Create an isolated environment and run:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Datasets (download locally; not stored in GitHub)

To keep the repository small and avoid GitHub web-upload limits, dataset files are intentionally excluded from version control. **The existing `results/` directory is preserved; no result file was removed or modified for this GitHub-ready package.**

Download the public datasets from their original providers:

- **UCI Human Activity Recognition Using Smartphones (UCI HAR):** https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- **Case Western Reserve University Bearing Data Center (CWRU):** https://engineering.case.edu/bearingdatacenter/download-data-file
- **Intel Berkeley Research Lab sensor data:** https://db.csail.mit.edu/labdata/labdata.html

Expected local layout before preprocessing:

```text
data/
├── raw/
│   └── cwru/              # CWRU .mat files named by file ID, e.g. 97.mat
├── extracted/
│   └── uci_har/
│       └── UCI HAR Dataset/
└── processed/             # generated locally; ignored by Git
```

Generate the processed arrays used by the experiment code:

```bash
python -m src.preprocess_public --datasets uci_har cwru
python -m src.preprocess_intel --source PATH/TO/data.txt.gz --output data/processed/intel_sensor.npz
```

The preprocessing code documents the exact split, feature construction, and leakage controls. Dataset files remain local and are ignored by `.gitignore`.

## One run

The following command reproduces one 300-round proposed-method run on UCI HAR:

```bash
python -m src.experiment \
  --data data/processed/uci_har.npz \
  --method proposed --variant full \
  --rho 0.3 --sigma 0.6 --eta-f 0.0005 \
  --seed 11 --rounds 300 --clients 80 --clients-per-round 24 \
  --output results/example/uci_har_proposed_11
```

Valid methods are `fedavg`, `fedprox`, `fmtl`, `topksgd`, `scaffold`, and
`proposed`. Valid proposed-method variants are `full`, `no_scheduler`,
`no_taskaware`, `no_ef`, and `no_freeze`.

## Restore raw results

The GitHub-ready package does not discard any experiment logs. It bundles the many small raw-result files into one lossless archive. Before re-running the analysis, restore them with:

```bash
bash scripts/unpack_results.sh
```

This recreates the original `results/full`, `results/v2`, `results/extended`, and other archived run directories.

## Analysis

After restoring the declared runs, regenerate all aggregate tables and sensitivity
figures with:

```bash
python -m src.analyze --results results/full \
  --proposed-results results/v2/main --output results/analysis
python -m src.analyze_extended
```

The extended analyzer accepts only 300-round runs with seeds 11, 22, 33, 44,
and 55, 24 selected clients, and the declared population. This prevents smoke
tests or obsolete scheduler runs from entering the paper.

## Archival DOI checklist

Before depositing the release in Zenodo or another DOI repository:

1. choose an open-source license approved by every rights holder;
2. replace the placeholder repository/DOI in `CITATION.cff`;
3. freeze the environment and archive hash;
4. verify third-party raw-data redistribution terms;
5. create a versioned release and then insert the assigned DOI into the paper.

No license is imposed by this draft archive because licensing is an authorial
rights decision. The included `CITATION.cff` and `.zenodo.json` provide the
remaining metadata scaffold.
