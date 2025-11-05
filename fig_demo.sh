#Training
python src/main.py --model-name GEDIOT-small --dataset IMDB --model-epoch-start 0 --model-epoch-end 2 --model-train 1 --demo
python src/main.py --model-name GedGNN-small --dataset IMDB --model-epoch-start 0 --model-epoch-end 2 --model-train 1 --demo
#Testing
python src/main.py --model-name GEDIOT-small --dataset IMDB --model-epoch-start 2 --model-epoch-end 2 --model-train 0 --postk 4 --demo
python src/main.py --model-name GEDHOT-small --dataset IMDB --model-epoch-start 2 --model-epoch-end 2 --model-train 0  --GW --postk 4 --demo
python src/main.py --model-name GedGNN-small --dataset IMDB --model-epoch-start 2 --model-epoch-end 2 --model-train 0 --postk 4 --demo