import layers
import numpy as np
from collections import Counter



def test(inputs):
    x = inputs[0]
    y = inputs[1]
    if (2*x  + y > 5):
        return 1
    else: 
        return 0


def listTest():
    data = np.random.random((100, 2))
    labels = np.array(list(map(test, data)))
    lays = []
    for i in range(5):
        lays.append(layers.denseLayer(2, 2))
        lays.append(layers.sigmoid())
    out = layers.sumSquareError()
    
    for iteration in range(100):
        IterationError = 0
        for datum, label in zip(data, labels):
            res = [lays[0].eval(datum)]
            for i in lays:
                res.append(i.eval(res[-1]))
            E = out.eval(res[-1], label)
            IterationError += E
            dE = out.inputDerivatives(res[-1], label)
            for i in range(len(lays)-1, -1,-1):
                layer = lays[i]
                dE = layer.update(res[i], dE)
        print(IterationError)


def lstmTest():
    data = np.random.random((100, 2))
    labels = np.array(list(map(test, data)))
    lstm = layers.lstmLayer(2, 1)
    out = layers.sumSquareError()
    for iteration in range(100):
        IterationError = 0.0
        for datum, label in zip(data, labels):
            res = []
        
            for i in range(10):
                res.append(lstm.eval(datum))
            E = out.eval(res[-1], label)
            IterationError += E
            dE = out.inputDerivatives(res[-1], label)
            Dinput, Dprev = lstm.setUpdate(datum, dE)
            for i in range(9, 0, -1):
                Dinput, Dprev = lstm.setUpdate(datum, Dprev)
        lstm.doUpdate()
        print("iteration", iteration, ": ", IterationError)


def convTest2():
    data = np.random.random((100, 2,1))
    labels = np.array(list(map(test, data)))
    conv1 = layers.convLayer(3, 3, 1)
    dense = layers.denseLayer(6, 1)
    cost = layers.sumSquareError()
    for i in range(100):
        itError = 0
        for datum, label in zip(data, labels):
            val1 = conv1.eval([datum])
            #print("Val1", val1)
            shmush = val1.reshape((6))
            #print("Shmush", shmush)
            val2 = dense.eval(shmush)
            error = cost.eval(val2, label)
            #print("Error", error)
            itError += error
            Derror = cost.setUpdate(val2, label)
            #print("Derror", Derror)
            Ddense = dense.setUpdate(shmush, Derror)
            #print("DDense", Ddense)
            DunShmush = Ddense.reshape((3, 2,1))
            #print("DUnShmush", DunShmush)
            D1 = conv1.setUpdate([datum], DunShmush)
            conv1.doUpdate()
            dense.doUpdate()
        print(i, ": ", itError)

def convTest3():
    data = np.random.random((10, 2,1))
    labels = np.array(list(map(test, data)))
    inp = layers.convLayer(3, 3, 1)
    conv1 = layers.convLayer(3, 3, 3)
    dense = layers.denseLayer(6, 1)
    cost = layers.sumSquareError()
    for i in range(100):
        itError = 0
        for dat, label in zip(data, labels):
            datum = inp.eval([dat])
            val1 = conv1.eval(datum)
            #print("Val1", val1.shape)
            shmush = val1.reshape((6))
            #print("Shmush", shmush)
            val2 = dense.eval(shmush)
            error = cost.eval(val2, label)
            #print("Error", error)
            itError += error
            Derror =  cost.setUpdate(val2, label)
            #print("Derror", Derror)
            Ddense = dense.setUpdate(shmush, Derror)
            #print("DDense", Ddense)
            DunShmush = Ddense.reshape((3, 2, 1))
            #print("Unshmush", DunShmush.shape)
            #print("DUnShmush", DunShmush)
            D1 = conv1.setUpdate(datum, DunShmush)
            DInp = inp.setUpdate([dat], D1)
        conv1.doUpdate(lr= .001)
        dense.doUpdate(lr = .001)
        inp.doUpdate(lr = .001)
        print(i, ": ", itError)

def denseTest():
    data = np.random.random((100, 2))
    labels = np.array(list(map(test, data)))
    dense = layers.denseLayer(2, 1)
    sig = layers.sigmoid()
    cost = layers.sumSquareError()
    for i in range(1000):
        itError = 0
        for datum, label in zip(data, labels):
            value_pre= dense.eval(datum) 
            value = sig.eval(value_pre)
            error = cost.eval(value, label)
            itError += error
            Derror = cost.setUpdate(value, label)
            Derror = sig.setUpdate(value_pre, Derror)
            dense.setUpdate(datum, Derror)
        print("Iteration ", i, " Error : ",itError)
        dense.doUpdate()
    print(dense.weights)
    print(dense.bias)