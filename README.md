# PI_CRNN_RBC_2026

Supplemental codes for "A Physics-Informed Spatiotemporal Deep Learning Framework for Turbulent Systems".

There are the following scripts in the `code` folder:

1) `functions.py` contains helper functions to load and preprocess the data
2) `cae_model.py` contains code to run the CAE; the script `cae_results.py' reproduces Table 1 and Figure 2.
3) `pi_crnn_model.py` contains code to run the physics-informed spatiotemporal model; the script `pi_crnn_results.py' reproduces Table 2 and Figure 3.
4) `uq.py` contains code to reproduce predictions intervals using conformal method

The `code` folder also contains a reproducible version able to run on a desktop machine. The files are similar and are in the `for_desktop` directory inside of `code`.

The DNS data for Rayleigh-Benard Convection used in this work is in the data folder, along with files containing the coordinates and physical constants. These can be reproduced from [this](https://git.uwaterloo.ca/SPINS/SPINS_main) public repository.

To reproduce the results from the manuscript, the user should first run `cae_model.py` to train the CAE portion of the model. The CAE will subsequently be saved as `cae.keras`. Then, run `pi_crnn_model.py` to train the proposed PI-CRNN approach. The files `cae_results.py' and `pi_crnn_results.py' reproduce the Tables and Figures in the main manuscript. The `uq.py` script can be run after training the model to obtain predictions intervals and verify proper coverage.
