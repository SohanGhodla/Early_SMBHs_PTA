
# Whats available?

This repository contains two directories:

1. The `Dark_stars_PTA`directory contains the code used for calculating the stochastic gravitational wave background (SGWB) resulting from early seeding of super massive black holes (SMBHs). The seeding mechanism is assumed to be via the collapse of *super massive dark stars* but the code works independent of the assumed seeding mechanism as long as some mechanism can produce the seeds.

2. The `Traditional_means_PTA` directory contains the code used for calculating the SGWB when one simply uses a black hole - halo mass relation to populate dark matter halos. This method does not care how and when the SMBHs were seeded.


# Installation instructions

This code was built in `python 3.12`. For best outcome use `pyenv` to run this program in the same **Python version**.

1. Begin by creating a virtual environment and then install dependencies uisng 

```bash
python -r requirements.txt
```