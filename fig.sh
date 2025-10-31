#Training
python src/main.py --model-name GEDIOT-small --dataset IMDB --model-epoch-start 0 --model-epoch-end 20 --model-train 1
python src/main.py --model-name GedGNN-small --dataset IMDB --model-epoch-start 0 --model-epoch-end 20 --model-train 1
#Testing
python src/main.py --model-name GEDIOT-small --dataset IMDB --model-epoch-start 20 --model-epoch-end 20 --model-train 0 
python src/main.py --model-name GEDHOT-small --dataset IMDB --model-epoch-start 20 --model-epoch-end 20 --model-train 0 --GW
python src/main.py --model-name GedGNN-small --dataset IMDB --model-epoch-start 20 --model-epoch-end 20 --model-train 0