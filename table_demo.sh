#Training
python src/main.py --model-name GEDIOT --dataset AIDS --model-epoch-start 0 --model-epoch-end 2 --model-train 1 --demo
python src/main.py --model-name GEDIOT --dataset Linux --model-epoch-start 0 --model-epoch-end 2 --model-train 1 --demo
python src/main.py --model-name GEDIOT --dataset IMDB --model-epoch-start 0 --model-epoch-end 2 --model-train 1 --demo
#Testing
python src/main.py --model-name GEDIOT --dataset AIDS --model-epoch-start 2 --model-epoch-end 2 --model-train 0 --path --postk 4 --demo
python src/main.py --model-name GEDHOT --dataset AIDS --model-epoch-start 2 --model-epoch-end 2 --model-train 0 --GW --path --postk 4 --demo
python src/main.py --model-name GEDGW --dataset AIDS --GW --path --postk 4 --demo
python src/main.py --model-name GEDIOT --dataset Linux --model-epoch-start 2 --model-epoch-end 2 --model-train 0 --path --postk 4 --demo
python src/main.py --model-name GEDHOT --dataset Linux --model-epoch-start 2 --model-epoch-end 2 --model-train 0 --GW --path --postk 4 --demo
python src/main.py --model-name GEDGW --dataset Linux --GW --path --postk 4 --demo
python src/main.py --model-name GEDIOT --dataset IMDB --model-epoch-start 2 --model-epoch-end 2 --model-train 0 --path --postk 4 --demo
python src/main.py --model-name GEDHOT --dataset IMDB --model-epoch-start 2 --model-epoch-end 2 --model-train 0 --GW --path --postk 4 --demo
python src/main.py --model-name GEDGW --dataset IMDB --GW --path --postk 4 --demo