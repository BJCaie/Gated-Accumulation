# Gated-Accumulation
Repository for code from the paper "Gating an expectation of sensory timing balances anticipation with evidence accumulation". 

Functions necessary to run notebooks are stored in folder 'Scripts'

Data is available for download at OSF https://osf.io/6zjkd/ 

To run data analysis, download the dataset, save to a local folder, and replace 'path' variable in notebooks with the corresponding local path

Model fits are computationally expensive, so best-fititng model parameters are also stored with OSF. However, code for fitting the model is included and can be done locally.

External packages required to reproduce the results are found in requirements.txt

Notebooks:

01_Behavioural_Data_Analysis.ipynb 
-Runs behavioural data analaysis on free choice saccade task and generates data analysis figures from the paper 

02_Simulate_Gated_Accumulator.ipynb
-Visualizes simulations of gated accumulator model

03_Grouped_Model_Fitting.ipynb
-Fits models to group-average data and visualizes model output

04_Individual_Model_Comparison.ipynb
-Fits gated and extended LATER model to individual participants, plots model errors

Scripts:
behaviour.py
- contains necessary functions for processing, analyzing, and plotting behavioural analysis of free choice saccade experiment.
gating.py
- contains model simulation and fitting functions
