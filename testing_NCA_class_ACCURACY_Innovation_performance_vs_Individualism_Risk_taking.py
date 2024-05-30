# Import required packages
import pandas as pd
import numpy as np
import matplotlib
from src.NecessaryConditionAnalysis.main import NCA 

# Load dataset
data = pd.read_csv('src/NecessaryConditionAnalysis/datasets/NCA_example_Individualism_Risk_taking_Innovation_performance_Dul_2016.csv', index_col=0)
X_columns = ["Individualism"]
y_column = "Innovation performance"

# Initialize NCA instance (instantiate NCA class an assign it to variable NCA_model)
# By default, it uses the CE-FDH, CR-FDH ceilings, as well as OLS
NCA_model = NCA(ceilings=["ce-fdh","cr-fdh","ols"])
# Define Condition(s)/Determinants (x) and Outcome (y) in the dataset (data)
# and fit the model with the columns X and y on the dataset (data)
NCA_model.fit(X=X_columns, y=y_column, data=data)

################ TESTING ###############################################

# Test NCA model accuracy for the selected ceiling line technique(s)
print("Accuracy:")
print(NCA_model.accuracy_)
print("")
print(NCA_model.accuracy_["ce-fdh"].iloc[0] == 1)
print(NCA_model.accuracy_["cr-fdh"].iloc[0] == 0.9285714285714286)

# Print NCA scatterplot for given ceilings and specified determinant (X)
NCA_model.plot(X_columns[0])
