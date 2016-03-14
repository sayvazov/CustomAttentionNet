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
            Dinput, Dprev, Dweights, Dbiases = lstm.update(datum, dE)
            for i in range(9, 0, -1):
                Dinput, Dprev, Dweight, Dbias = lstm.update(datum, Dprev)
                Dweights += Dweight
                Dbiases += Dbias
            #print("Weights ",lstm.weights)
            #print("Delta Weights", Dweights)
            print("bias ",lstm.bias)
            print("Delta bias", Dbiases)
            lstm.weights += 0.1*Dweights
            lstm.bias += 0.1*Dbiases
            print("iteration i: ", IterationError)

data = np.random.random((100, 2))
labels = np.array(list(map(test, data)))

arr = [[1,2],[3,4]]

lstmTest()


#conv = layers.convLayer(3, 1, weights=np.array([[[.1,.2,.3],[.4,.5,.6], [.7,.8,.9]]]), bias = [0])
#testData = np.array([[0.0,1,], [4,5]])
#outputDers = np.array([[[1,1], [2,1]]])
#print(conv.eval(testData))
#print(conv.weightDerivatives(testData, outputDers))
#print(conv.biasDerivative(testData, outputDers))
#print(conv.inputDerivatives(testData, outputDers))

#dense = layers.denseLayer(2, 1)
#sig = layers.sigmoid()
#cost = layers.sumSquareError()
#for i in range(1000):
#    itError = 0
#    for datum, label in zip(data, labels):
#        value_pre= dense.eval(datum) 
#        value = sig.eval(value_pre)
#        error = cost.eval(value, label)
#        itError += error
#        Derror = cost.inputDerivatives(value, label)
#        Derror = sig.inputDerivatives(value_pre, Derror)
#        DW = dense.weightDerivatives(datum, Derror)
#        Dbias = dense.biasDerivative(Derror)
#        dense.weights += 0.1*DW
#        dense.bias  += 0.1* Dbias
#    print("Iteration ", i, " Error : ",itError)
#print(dense.weights)
#print(dense.bias)





