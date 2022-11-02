import functions as fc
import pandas as pd
import numpy as np
import scipy
import torch.nn as nn
from model import Model

data, text = fc.getData()

print("Saving Arrays")
data.to_csv("df.csv")
text.to_csv("text.csv")
print("Done!")
