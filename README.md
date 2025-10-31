# GEDOT Reproducibility Guide
This repo aims to reproduce the results of [GEDOT](https://dl.acm.org/doi/10.1145/3709673). The official implementation of GEDOT is available at this [repo](https://github.com/chengqihao/GED-via-Optimal-Transport). The following instructions will guide you through the process of reproducing our experiments in the paper.

- **GEDIOT** — a **learning-based** method.
- **GEDGW** — an **unsupervised** method.
- **GEDHOT** — an **ensemble** approach that integrates GEDIOT and GEDGW.

## Getting Started
### Requirements
All implementations are based on Python 3.9.

```
dgl                       1.0.2
pot                       0.8.2
networkx                  3.1
numpy                     1.26.4
scipy                     1.12.0
pytorch                   2.2.2+cpu
pyg                       2.5.2 
torchvision               0.17.2
texttable                 1.6.4
tqdm                      4.65.0
```
### Environment Setup

A shell script `setup_gedot_env.sh` is provided to create the `gedot_repro` environment and install all dependencies using Conda.

```bash
sh setup_gedot_env.sh
```
Activate the environment:

```bash
conda activate gedot_repro
```

### Datasets

The datasets utilized in this study are **AIDS**, **Linux**, and **IMDB**. These datasets are provided in the directory `./json_data`. The ground-truth GEDs and node matchings are stored in the file `TaGED.json`.

## Experiments

#### Results Storage

Results of **Table 3–5** and **Figure 8** will be saved under:

```text
./
├── tab3/        # Results of Table 3
│   ├── AIDS.csv
│   ├── Linux.csv
│   └── IMDB.csv
├── tab4/        # Results of Table 4 (Graph Edit Path)
│   ├── AIDS.csv
│   ├── Linux.csv
│   └── IMDB.csv
├── tab5/        # Results of Table 5
│   ├── AIDS.csv
│   ├── Linux.csv
│   └── IMDB.csv
└── fig8/        # Results of Figure 8
    └── IMDB.csv
```

### Training

- **GEDIOT** and **GEDHOT** require **20 epochs** of training GEDIOT.

- For **GEDGW**, training is not required (see *Testing* section).

```bash
sh training.sh
```
Our code can also reproduce baseline results by setting the argument `model-name` to `GedGNN`, `TaGSim`, `GPN` and `SimGNN`
### Testing

To reproduce the results in **Table 3, 4, and 5**, run:

```bash
sh table.sh
```

Note that the argument `–path` in `table.sh` determines whether to generate the Graph Edit Path (results shown in Table 4).

#### Result of Baselines

We provide a shell script `baseline.sh` to reproduce baseline results in **Table 3–5**:

```bash
sh baseline.sh
```

#### More Results of Figure 8
To reproduce results in **Figure 8** (methods with the `-small` suffix), run:
```bash
sh fig.sh
```


