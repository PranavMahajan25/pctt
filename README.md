# Predictive coding through time


## 1. Description
Repository for experiments with the predictive coding through time (PCTT) algorithm


## 2. Installation
To run the code, you should first install [Anaconda](https://www.anaconda.com/) or [Miniconda](https://conda.io/miniconda.html) (preferably the latter), 
and then clone this repository to your local machine.

Once these are installed and cloned, you can use the appropriate `.yml` file to create a conda environment. 
For Ubuntu or Mac OS, open a terminal (for Windows, open the Anaconda Prompt), go to the directory where you cloned the repo and then enter:

1. `cd pctt`
2. `conda env create -f environment.yml`  
3. `conda activate temporalenv`
4. `pip install -e .`  

## 3. Use
As an example, see how to use the code in `sequential_memory.py`. To run it, navigate to the repo directory and then enter:

`python sequential_memory.py`

Once you run these commands, a directory named `results` will be created to store all the data and figures collected from the experiments.

More to follow.