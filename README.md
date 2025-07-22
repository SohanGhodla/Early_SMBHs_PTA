
This repository contains the code to recreate the plots in the paper arXiv: [2507.06163](https://arxiv.org/abs/2507.06163)

 - The `src`directory contains the code used for calculating the stochastic gravitational wave background (SGWB) resulting from early seeding of super massive black holes (SMBHs). The seeding mechanism is assumed to be via the collapse of *super massive dark stars* but the code works independent of the assumed seeding mechanism as long as some mechanism can produce the seeds.

 - The `ipynb` files contain the plotting code.

 - The `sshRun.py` file is provided for batching the job on a cluster.

 - The `PTA_data` directory contain PTA observations.

 - The `Figure` directory contains the plots used in the paper.

 - The `outputForPlots` directory contain the csv files generated via ssh runs that are then plotted in the `plotMain.ipynb`.


### Installation instructions

This code was built in `python 3.12`. For best outcome use `pyenv` to run this program in the same **Python version**. Begin by creating a virtual environment and then install dependencies uisng 

```bash
python -r requirements.txt
```