"""
Use this testing file to test for known values of Accuracy and 
Ceiling Zone size 

"""

# Import required packages
import pandas as pd
import numpy as np
from src.NecessaryConditionAnalysis.main import NCA 

# Load dataset
data = pd.read_csv('src/NecessaryConditionAnalysis/datasets/NCA_example_Individualism_Risk_taking_Innovation_performance_Dul_2016.csv', index_col=0)
X_columns = ["Individualism","Risk taking"]
y_column = "Innovation performance"

# Initialize NCA instance (instantiate NCA class an assign it to variable NCA_model)
NCA_model = NCA(ceilings=["ce-fdh", "cr-fdh", "ols", "cols"])

# Define Condition(s)/Determinants (x) and Outcome (y) in the dataset (data)
# and fit the model with the columns X and y on the dataset (data)
NCA_model.fit(X=X_columns, y=y_column, data=data)

################ TESTING ###############################################
"""
This test is based on the results from the analysis shown in:
Dul, J. (2016). Necessary Condition Analysis (NCA): Logic and methodology
 of “necessary but not sufficient” causality.. ~Organizational Research Methods~, 
 19(1), 10-52.. https://doi.org/10.1177/1094428115584005
"""

# Test NCA model accuracy for the selected ceiling line technique(s)
print("")
print("Accuracy:")
print(NCA_model.accuracy_)
print("")
print("Tests (for individualism column):")
print("CE-FDH:",NCA_model.accuracy_["ce-fdh"].iloc[0] == 1)
print("CR-FDH:",round(NCA_model.accuracy_["cr-fdh"].iloc[0],2) == 0.93)
print("COLS:",NCA_model.accuracy_["cols"].iloc[0] == 1)


# Test NCA model size of ceiling zone
print("")
print("Size of ceiling zone :")
print(NCA_model.ceiling_size_)
print("")
print("Tests (for individualism column):")
print("CE-FDH:",round(NCA_model.ceiling_size_["ce-fdh"].iloc[0],2) == 6466.8)
print("CR-FDH:",round(NCA_model.ceiling_size_["cr-fdh"].iloc[0],2) == 4772.54)
print("COLS:",round(NCA_model.ceiling_size_["cols"].iloc[0],2) == 2430.65)


# Test NCA model effects
print("")
print("Effect sizes:")
print(NCA_model.effects_)
print("")
print("Tests (for individualism column)")
print("CE-FDH:",round(NCA_model.effects_["ce-fdh"].iloc[0],2) == 0.42)
print("CR-FDH:",round(NCA_model.effects_["cr-fdh"].iloc[0],2) == 0.31)