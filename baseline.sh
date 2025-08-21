baselines="GPN SimGNN TaGSim GedGNN"
datasets="AIDS Linux IMDB"
for method in $baselines; do
    for dataset in $datasets; do
        python src/main.py --model-name $method --dataset $dataset --model-epoch-start 0 --model-epoch-end 20 --model-train 1
    done
done
# Methods: GPN, SimGNN, TaGSim, GedGNN
for method in $baselines; do
    for dataset in $datasets; do
        python src/main.py --model-name $method --dataset $dataset --model-epoch-start 20 --model-epoch-end 20 --model-train 0
    done
done
# Methods: NOAH, Classic and GedGNN (Path) 
for dataset in $datasets; do
    python src/main.py --model-name NOAH --dataset $dataset --model-epoch-start 20 --model-epoch-end 20 --model-train 0
    python src/main.py --model-name GedGNN --dataset $dataset --model-epoch-start 20 --model-epoch-end 20 --model-train 0 --path
    python src/main.py --model-name Classic --dataset $dataset --greedy --path
done
