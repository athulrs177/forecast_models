# INSTALLATION
Python package to generate deterministic forecasts of rainfall amounts using Gamma Regression and CNN models. This code accompanies the article titled ``` Machine learning models for daily rainfall forecasting in Northern Tropical
Africa using tropical wave predictors``` submitted to ``` Weather and Forecasting```. A preprint is available in arxiv under: ```http://arxiv.org/abs/2408.16349```.

Install using pip: ```pip install git+https://github.com/athulrs177/forecast_models.git```.
If this causes issues, then you may try downloading the directory ```forecast_models```.

# RUNNING THE MODEL(S)
For running the example jupyter-notebook provided, you will need a machine with 13 logical CPUs and ~40 GB of memory in the case of Gamma Regression model and 40 logical CPUs and ~120 GB of memory in the case of CNN model. Weaker machines may also work provided enough memory to load the data/ with modified batch sizes, but performance may be poorer. In their current forms, these models are not optimized to run on GPUs, although perfromance gains may be attained. Note that the results of the CNN model may vary slightly on different machines.


