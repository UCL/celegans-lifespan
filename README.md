# celegans-lifespan
Machine learning models to predict lifespan of CElegans from pathology descriptors. 

## ML model comparison
The following script compares different linear and non-linear ML models for predicting CElegans lifespan (regression) from pathology features:

      python lifespan_regression_compare_models_per_day.py --control_path ../data/control_data_build_model.xlsx --mutant_path ../data/mutants_test_model_data.xlsx --out_dir /path/to/output/folder

![R2 Score](/figs/r2scores_all.png)  
<p align="left">
 Lifespan prediction model comparison (R2 score) </left>
</p>

![Mean Average Error](/figs/mae_scores_all.png)  
<p align="left">
 Lifespan prediction model comparison (MAE) </center>
</p>

The script also performs comparisons per day using multiple metrics (R2, MAE, RMSE, F1-score (>18 days), Acccuracy (>18 days), ...)

## Data analysis and feature importance 
The following script computes the feature importance (Mean Decrease in Impurity) for a Random Forrest Regression model together with additional analysis (correlation between features, scatter plots for prediction, etc.).

      python random_forest_scatter_plots_corr_matrix_fi.py --control_path ../data/control_data_build_model.xlsx --mutant_path ../data/mutants_test_model_data.xlsx --out_dir /path/to/output/folder

![Mean Decrease in Impurity](/figs/mdi_all.png)  
<p align="left">
 Feature importance for predicting lifespan </center>
</p>



![Correlation Matrix](/figs/corr_matrix.png)  
<p align="left">
 Feature correlations </center>
</p>


## **Requirements**

- Python3.6
- Numpy
- Pandas
- scikit-learn
- seaborn
