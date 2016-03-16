import layers
import numpy as np
from collections import Counter



def test(inputs):
    x = inputs[0]
    y = inputs[1]
    if (2*x  + y > 1.5):
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
    data = np.random.random((5, 2,1))
    labs = np.array(list(map(test, data)))
    labels = []
    count = 0
    print(labs)
    for lab in labs:
        if lab == 1:
            labels.append([0,1])
            count += 1
        else:
            labels.append([1,0])
    lstm = layers.lstmLayer(2, 2)
    out = layers.crossEntropy()
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
    data = np.random.random((100, 2,1))
    labs = np.array(list(map(test, data)))
    labels = []
    count = 0
    print(labs)
    for lab in labs:
        if lab == 1:
            labels.append([0,1])
            count += 1
        else:
            labels.append([1,0])
    print(count)

    inp = layers.convLayer(3, 3, 1)
    conv1 = layers.convLayer(3, 3, 3)
    conv2 = layers.convLayer(3,3,3)
    sig = layers.tanh()
    dense = layers.denseLayer(6, 2)
    soft = layers.softMax()
    cost = layers.sumSquareError()
    for i in range(3000):
        itError = 0
        epochCorrect = 0
        correct = False
        for dat, label in zip(data, labels):
            datum = inp.eval([dat])
            c1= conv1.eval(datum)
            c1s = sig.eval(c1)
            c2 = conv2.eval(c1s)
            c2s = sig.eval(c2)
            #print("Val1", val1.shape)
            shmush = c2s.reshape((6))
            #print("Shmush", shmush)
            val2 = dense.eval(shmush)
            sof = soft.eval(val2)
            #print(sof, label)
            if label[0] == 1 and sof[0] > sof[1]:
                correct = True
                epochCorrect+=1
                #print("Correct :", sof[0], ">", sof[1])
            elif label[1] ==1 and sof[1] > sof[0]:
                correct = True
                epochCorrect+=1
                #print("Correct :", sof[0], "<", sof[1])
            else:
                #print("Incorrect :", sof[0], " ", sof[1])
                pass
            error = cost.eval(sof, label)
            #print("Error", error)
            itError += error
            if True:
                Derror =  cost.setUpdate(sof, label)
                Dsof = soft.setUpdate(val2, Derror)
                #print("Derror", Derror)
                Ddense = dense.setUpdate(shmush, Dsof)
                #print("DDense", Ddense)
                DunShmush = Ddense.reshape((3, 2, 1))
                #print("Unshmush", DunShmush.shape)
                #print("DUnShmush", DunShmush)
                dc2s = sig.setUpdate(c2, DunShmush)
                dc2 = conv2.setUpdate(c1s, dc2s)
                dc1s = sig.setUpdate(c1, dc2)
                dc1 = conv1.setUpdate(datum,dc1s)
                dInp = inp.setUpdate([dat], dc1)
        conv1.doUpdate(lr= .01)
        conv2.doUpdate(lr = .01)
        dense.doUpdate(lr = .01)
        inp.doUpdate(lr = .01)
        print(i, ": ", itError, "Accuracy : ", epochCorrect)

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