# Create and activate the virtual environment
conda create -n gedot_repro python=3.9 -y
source activate gedot_repro

# Install PyTorch (CPU version)
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 cpuonly -c pytorch -y

# Install PyG (PyTorch Geometric)
conda install pyg==2.5.2 -c pyg -y

# Install POT library from conda-forge
conda install -c conda-forge pot==0.8.2 -y

# Install other dependencies
conda install pandas -y
pip install dgl==1.0.2 -f https://data.dgl.ai/wheels/repo.html
pip install scipy==1.12.0
pip install numpy==1.26.4
pip install networkx==3.1
pip install texttable==1.6.4
pip install tqdm==4.65.0