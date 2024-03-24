import pandas as pd
import random
import numpy as np

dataX = None
dataY = None
testX = None
testY = None
validate_ratio = 0.05

def load_data(file, state, out):
    global dataX, dataY, testX, testY

    df = pd.read_csv(file)
    print('Shape of', file, ":", df.shape)
    n = int(df.shape[0]*11/12)
    t = int(n*(1-validate_ratio))
    dataX = df.values[0:t, state]
    dataY = df.values[0:t, out]
    testX = df.values[t:n, state]
    testY = df.values[t:n, out]
    print('GGL data : Train', len(dataX), ', Validate', len(testX))


def load_samples(sample_size):
    slice = random.sample(range(len(dataX)), sample_size)
    return dataX[slice], dataY[slice]

def load_validate():
    return testX, testY
