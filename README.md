# fpd-default-classification
Predicting first payment default for loan applications 

This outlines the structure of the project files and provides instructions on how to run the model for the First Payment Default (FPD) prediction task.

### PROJECT FILES & DESCRIPTION

| File | Description |
|-------|-------------|
| **LucidIntel - FPD Model Workflow.ipynb** | Main Jupyter Notebook containing all steps: EDA, preprocessing, feature engineering, model training, and evaluation. |
| **final_model_pipeline.pkl** | Serialized pipeline (.pkl) including preprocessing steps and trained LightGBM model for predictions on new data. |
| **features_summary.csv** | Summary of final features used, with short descriptions and data types. |
| **model_utils.py** | Utility script with functions for data transformation and preprocessing. |
| **requirements.txt** | List of Python libraries and versions needed to run notebooks and scripts. |
| **input.json** | JSON file containing parameters used by the Python code (e.g., configuration settings). |
| **TE.pkl** | Serialized TargetEncoder object used for encoding categorical fields in data. |
| **FPD_Model_Report.pdf** | Comprehensive report outlining the entire model development workflow. |
| **data.csv** | Dataset used for building the model (anonymized). |
| **tree.png** | Visualization of the final decision tree (model interpretation aid). |
| **model_inference.ipynb** | Jupyter notebook to load the trained pipeline and generate predictions on new data. |


### HOW TO RUN THE MODEL ON NEW DATA
-	Set Up Your Environment. Ensure Python 3.8+ and dependencies are installed (first cell of notebook will install all libraries. Comment out that code if already installed. 
-	Enter path (or filename if in same folder) of the prediction dataset in “file_name” field of input.json
-	Run Inference notebook – model_inference.ipynb 

![image](https://github.com/user-attachments/assets/e2a73225-563d-470e-9a90-752207831bcf)
