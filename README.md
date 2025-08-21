# GEDOT Reproducibility Guide
This repo aims at reproducing the results of [GEDOT](https://dl.acm.org/doi/10.1145/3709673). The official implementation of GEDOT is available at this [repo](https://github.com/chengqihao/GED-via-Optimal-Transport). The following instruction will guide you through the process of reproducing our experiments in the paper.

- **GEDIOT** — a **learning-based** method.
- **GEDGW** — an **unsupervised** method.
- **GEDHOT** — an **ensemble** approach that integrates GEDIOT and GEDGW.

## Getting Started
### Requirements
All codes are implemented in python3.9

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
### Install Environment
```bash
conda create -n gedot python=3.9
conda activate gedot
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 cpuonly -c pytorch
conda install pyg=2.5.2 -c pyg
conda install -c conda-forge pot=0.8.2
conda install pandas
pip install dgl==1.0.2 -f https://data.dgl.ai/wheels/repo.html
pip install scipy==1.12.0
pip install numpy==1.26.4
pip install networkx==3.1
pip install texttable==1.6.4
pip install tqdm==4.65.0
```
### Datasets

The datasets utilized in this study are **AIDS**, **Linux**, and **IMDB**. These datasets are provided in the directory `./json_data`. The ground-truth GEDs and node matchings are stored in the file `TaGED.json`

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
python src/main.py --model-name GEDIOT --dataset AIDS --model-epoch-start 0 --model-epoch-end 20 --model-train 1
python src/main.py --model-name GEDIOT --dataset Linux --model-epoch-start 0 --model-epoch-end 20 --model-train 1
python src/main.py --model-name GEDIOT --dataset IMDB --model-epoch-start 0 --model-epoch-end 20 --model-train 1
```
Our code can also reproduce baseline results by setting the argument `model-name` to `GedGNN`, `TaGSim`, `GPN` and `SimGNN`
### Testing

To reproduce the results in **Table 3, 4, and 5**, run:

```bash
python src/main.py --model-name GEDIOT --dataset AIDS --model-epoch-start 20 --model-epoch-end 20 --model-train 0 --path
python src/main.py --model-name GEDHOT --dataset AIDS --model-epoch-start 20 --model-epoch-end 20 --model-train 0 --GW --path
python src/main.py --model-name GEDGW --dataset AIDS --GW --path
python src/main.py --model-name GEDIOT --dataset Linux --model-epoch-start 20 --model-epoch-end 20 --model-train 0 --path
python src/main.py --model-name GEDHOT --dataset Linux --model-epoch-start 20 --model-epoch-end 20 --model-train 0 --GW --path
python src/main.py --model-name GEDGW --dataset Linux --GW --path
python src/main.py --model-name GEDIOT --dataset IMDB --model-epoch-start 20 --model-epoch-end 20 --model-train 0 --path
python src/main.py --model-name GEDHOT --dataset IMDB --model-epoch-start 20 --model-epoch-end 20 --model-train 0 --GW --path
python src/main.py --model-name GEDGW --dataset IMDB --GW --path
```

Note that the argument `–path`  is used to determine whether to generate Graph Edit Path (Results on Table 4).

#### Result of Baselines

We provide a shell script `baseline.sh` to reproduce baseline results in **Table 3–5**:

```bash
sh baseline.sh
```

#### More Results of Figure 8
To reproduce results in **Figure 8** (methods with the `-small` suffix), run:
```bash
#Training
python src/main.py --model-name GEDIOT-small --dataset IMDB --model-epoch-start 0 --model-epoch-end 20 --model-train 1
python src/main.py --model-name GedGNN-small --dataset IMDB --model-epoch-start 0 --model-epoch-end 20 --model-train 1
```
```bash
#Testing
python src/main.py --model-name GEDIOT-small --dataset IMDB --model-epoch-start 20 --model-epoch-end 20 --model-train 0 
python src/main.py --model-name GEDHOT-small --dataset IMDB --model-epoch-start 20 --model-epoch-end 20 --model-train 0 --GW
python src/main.py --model-name GedGNN-small --dataset IMDB --model-epoch-start 20 --model-epoch-end 20 --model-train 0
```

