# Import required packages
import pandas as pd
import numpy as np
import matplotlib
from src.NecessaryConditionAnalysis.main import NCA 

# Load dataset
data = pd.read_csv('src/NecessaryConditionAnalysis/datasets/NCA-MOOC-2005.csv', index_col=0)
X_columns = ["Vaccination","Basic education"]
y_column = "Life expectancy"
print(data.head())
print("")

# Initialize NCA instance (instantiate NCA class an assign it to variable NCA_model)
# By default, it uses the CE-FDH, CR-FDH ceilings, as well as OLS
NCA_model = NCA(ceilings=["ce-fdh","cr-fdh","ols"])
print("Ceilings:")
print(NCA_model.ceilings)
print("")

# Define Condition(s)/Determinants (x) and Outcome (y) in the dataset (data)
# and fit the model with the columns X and y on the dataset (data)
NCA_model.fit(X=X_columns, y=y_column, data=data)

# Print NCA model effects sizes
print("Effects:")
print(NCA_model.effects_)
print("")

# Print NCA bottleneck table for the selected ceiling line technique(s)
print("Bottleneck table:")
print(NCA_model.bottleneck(ceiling='cr-fdh',bottleneck_type = "percentage"))
print("")

# Print NCA scatterplot for given ceilings and specified determinant (X)
#NCA_model.plot(X_columns[1])
print("")

# Print NCA model accuracy for the selected ceiling line technique(s)
print("Accuracy:")
print(NCA_model.accuracy_)
print("")

# Print NCA condition inefficiency for the selected ceiling line technique(s)
print("Condition inefficiency:")
print(NCA_model.condition_inefficiency_)
#print(NCA_model.condition_inefficiency_point_)
print("")

# Print NCA outcome inefficiency for the selected ceiling line technique(s)
print("Outcome inefficiency:")
print(NCA_model.outcome_inefficiency_)
#print(NCA_model.outcome_inefficiency_point_)
print("")

print(pd.__version__)
print(np.__version__)
print(matplotlib.__version__)