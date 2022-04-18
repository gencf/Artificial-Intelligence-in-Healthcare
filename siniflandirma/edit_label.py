import pandas as pd

import os
labels = {}
for i,f in enumerate(os.listdir("rotated")):
    if f[:2] == "IN":
        labels[f] = 0
    elif f[:2] == "IS":
        labels[f] = 1
    elif f[:2] == "KA":
        labels[f] = 2

pd.DataFrame(list(labels.items()),columns = ['column1','column2']).to_csv("labels_rotated.csv")
