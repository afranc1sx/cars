import pandas as pandahelper

#load the dataset
attackerData = pandahelper.read_csv('C:/Users/a6shl/downloads/DoS_dataset.csv') 

#
print(attackerData.head())
attackerData.dropna(inplace=True)