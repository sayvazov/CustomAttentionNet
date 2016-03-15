import numpy as np
from functools import reduce

learningRate = 0.1

class layer: 
    def __init__():
        return
    def eval(self, input):
        return input
    def inputDerivatives(self, input, outputDerivatives):
        return np.ones_like(input)
    def setUpdate(self, input, outputDerivatives):
        return self.inputDerivatives(input, outputDerivatives)
    def doUpdate(self):
        return
        


class weightLayer(layer):
    def weightDerivatives(self, input, outputDerivatives):
        return 0
    def biasDerivative(self, input, outputDerivatives):
        return 0
    def setUpdate(self, input, outputDerivatives):
        dW = self.weightDerivatives(input, outputDerivatives)
        dB = self.biasDerivative(input, outputDerivatives)
        self.weightUpdates += dW
        self.biasUpdates += dB
        return self.inputDerivatives(input, outputDerivatives)
    def doUpdate(self):
        self.weights += learningRate * self.weightUpdates
        self.bias += learningRate * self.biasUpdates
        self.weightUpdates.fill(0)
        self.biasUpdates.fill(0)

class combineLayer(layer):
    pass

class resultLayer(layer):
    def inputDerivatives(self, guess, answer):
        return np.ones_like(guess)


class lstmLayer(weightLayer):
    def __init__(self, inputSize, outputSize, weights=None, bias=None):
        if weights == None:
            Wi = np.random.normal(size=(inputSize, inputSize))
            Wf = np.random.normal(size=(inputSize, inputSize))
            Wc = np.random.normal(size=(inputSize, inputSize))
            Wo = np.random.normal(size=(inputSize, inputSize))
            Ui = np.random.normal(size=(inputSize, inputSize))
            Uf = np.random.normal(size=(inputSize, inputSize))
            Uc = np.random.normal(size=(inputSize, inputSize))
            Uo = np.random.normal(size=(inputSize, inputSize))
            
            self.weights = np.array([Wi, Wf, Wc, Wo, Ui, Uf, Uc, Uo])
        else:
            self.weights = weights
        if bias == None:
            bi = np.random.normal(size=inputSize)
            bf = np.random.normal(size=inputSize)
            bc = np.random.normal(size=inputSize)
            bo = np.random.normal(size=inputSize)
            self.bias = np.array([bi, bf, bc, bo])
        else: 
            self.bias = bias
        self.cell = np.zeros(inputSize)
        self.hidden = np.zeros(inputSize)
        self.prevHidden = np.zeros(inputSize)
        self.sig = sigmoid()
        self.tan = tanh()
        self.weightUpdates = np.zeros_like(self.weights)
        self.biasUpdates = np.zeros_like(self.bias)
        return 

    def eval(self, input):
        self.inputSum       = np.dot(self.weights[0],input) + np.dot(self.hidden, self.weights[4]) + self.bias[0]
        self.forgetSum      = np.dot(self.weights[1],input) + np.dot(self.hidden, self.weights[5]) + self.bias[1]
        self.candidateSum   = np.dot(self.weights[2],input) + np.dot(self.hidden, self.weights[6]) + self.bias[2]
        self.outputSum      = np.dot(self.weights[3],input) + np.dot(self.hidden, self.weights[7]) + self.bias[3]

        self.inputGate  = self.sig.eval(self.inputSum) #
        self.forgetGate = self.sig.eval(self.forgetSum) #
        self.candidates = self.tan.eval(self.candidateSum) #
        self.outputGate = self.sig.eval(self.outputSum) #

        self.prevCell = self.cell #
        self.cell =  self.prevCell*self.forgetGate + self.candidates*self.inputGate#
        self.transformed = self.tan.eval(self.cell) #
        self.prevHidden = self.hidden #
        self.hidden = self.outputGate*self.transformed 

        return self.hidden

    def setUpdate(self, input, outputDerivatives):
        #calculate weight derivatives
        DOutputGate = self.transformed*outputDerivatives
        DTransformed = self.outputGate*outputDerivatives
        Dcell = self.tan.inputDerivatives(self.cell, DTransformed)
        DForgetGate = Dcell*self.prevCell
        DInputGate  = Dcell*self.candidates
        DprevCell   = Dcell*self.forgetGate
        DCandidates = Dcell*self.inputGate

        DOutputSum  = self.sig.inputDerivatives(self.outputSum, DOutputGate)
        DForgetSum      = self.sig.inputDerivatives(self.forgetSum, DForgetGate)
        DInputSum       = self.sig.inputDerivatives(self.inputSum, DInputGate)
        DCandidateSum   = self.tan.inputDerivatives(self.candidateSum, DCandidates)

        DWi = np.outer(DInputSum, input)
        DWf = np.outer(DForgetSum, input)
        DWc = np.outer(DCandidateSum, input)
        DWo = np.outer(DOutputSum, input)

        DUi = np.outer(DInputSum, self.prevHidden)
        DUf = np.outer(DForgetSum, self.prevHidden)
        DUc = np.outer(DCandidateSum, self.prevHidden)
        DUo = np.outer(DOutputSum, self.prevHidden)

        Dbi = DInputSum
        Dbf = DForgetSum
        Dbc = DCandidateSum
        Dbo = DOutputSum

        Dinput      = np.dot(DInputSum, self.weights[0]) + np.dot(DForgetSum, self.weights[1]) + np.dot(DCandidateSum, self.weights[2]) + np.dot(DOutputSum, self.weights[3])
        DprevHidden = np.dot(DInputSum, self.weights[4]) + np.dot(DForgetSum, self.weights[5]) + np.dot(DCandidateSum, self.weights[6]) + np.dot(DOutputSum, self.weights[7])

        Dweights = np.array([DWi, DWf, DWc, DWo, DUi, DUf, DUc, DUo])
        Dbias = np.array([Dbi, Dbf, Dbc, Dbo])

        self.weightUpdates += Dweights
        self.biasUpdates += Dbias

        return Dinput, DprevHidden




class denseLayer (weightLayer):
    def __init__(self, inputSize, outputSize, weights =None, bias = None):
        if weights == None:
            self.weights = np.random.normal(size=(inputSize, outputSize))
        else:
            self.weights = weights
        if self.weights.shape != (inputSize, outputSize):
            if np.shape( self.weights) == (outputSize, inputSize):
                self.weights = np.transpose(self.weights)
                print("Transposing Weights")
            else:
                self.weights = np.random.normal(size=(inputSize, outputSize))
                print("Weights size is incorrect. Defaulting to random weights")
        if bias == None:
            self.bias = np.random.normal(size=outputSize)
        else:
            self.bias = bias
        if outputSize == 1:
            return
        elif len(self.bias) != outputSize:
            self.bias = np.random.normal(size=outputSize)
            print("Bias was wrong size. Defaulting to random bias for ", self.name)
        self.weightUpdates = np.zeros_like(self.weights)
        self.biasUpdates = np.zeros_like(self.bias)
        return
    
    def eval(self, input):
        return np.dot(input, self.weights)+ self.bias

    def weightDerivatives(self, input, outputDerivatives):
        return np.outer(input, outputDerivatives)

    def inputDerivatives(self, input, outputDerivates):
        return np.dot(self.weights, outputDerivates)
    
    def biasDerivative(self, input, outputDerivatives):
        return outputDerivatives

class convLayer(weightLayer):
    def __init__(self, kernelSize, numFilters, weights=None, bias = None):
        self.kernelSize = kernelSize
        self.numFilters = numFilters
        if weights == None:
            self.weights = np.random.normal(size=( numFilters, kernelSize, kernelSize))
        else:
            self.weights = weights
        if bias == None: 
            self.bias = np.random.normal(size=numFilters)
        else:
            self.bias = bias
        self.weightUpdates = np.zeros_like(self.weights)
        self.biasUpdates = np.zeros_like(self.bias)
        return super.__init__(self)
    def eval(self, input):
        padded = np.pad(input,int(self.kernelSize/2) ,'constant', constant_values=0)
        print(padded) 
        res = np.array([np.zeros_like(input) for i in range(self.numFilters)])
        for filter in range(self.numFilters):
            for i in range(len(input)):
                for j in range(len(input[0])):
                    sum = 0.0
                    for row in range(self.kernelSize):
                        for col in range(self.kernelSize):
                            sum += padded[i+row, j+col]* self.weights[filter, row, col]
                    res[filter, i, j] = sum
        return res
    def weightDerivatives(self, input, outputDerivatives):
        padded = np.pad(input,int(self.kernelSize/2) ,'constant', constant_values=0)
        w = int(self.kernelSize/2)
        res = np.zeros_like(self.weights)
        for filter in range(self.numFilters):
            for row in range(self.kernelSize):
                for col in range(self.kernelSize):
                    sum = 0 
                    for i in range(len(input)):
                        for j in range(len(input[0])):
                            sum+=outputDerivatives[filter, i,j]*padded[i+row,j+col]
                    res[filter, row, col] += sum
        return res


    def biasDerivative(self, input, outputDerivatives):
        return np.sum(outputDerivatives, axis=0)

    def inputDerivatives(self, input, outputDerivatives):
        res = np.zeros_like(input)
        w = int(self.kernelSize/2)
        paddedDer = np.pad(outputDerivatives,((0,0), (w,w), (w,w)),'constant', constant_values=0)
       
        for filter in range(self.numFilters):
            for i in range(len(input)):
                for j in range(len(input[0])):
                    sum = 0
                    for row in range(self.kernelSize):
                        for col in range(self.kernelSize):
                            sum+= self.weights[filter, row, col]*paddedDer[filter, i-row+2*w, j-col+2*w]
                    res[i,j] = sum
        return res

class elemMulLayer(combineLayer):
    def __init__(self):
        return
    def eval(self, first, second):
        return first*second
    def inputDerivatives(self, first, second, outputDerivatives):
        return second*outputDerivatives, first*outputDerivatives
    
class sigmoid(layer):
    def __init__(self):
        pass
    def eval(self, input):
        return 1/(1 + np.exp(-1*input))
    def inputDerivatives(self, input, outputDerivatives):
        out = self.eval(input)
        return outputDerivatives*out*(1-out)

class tanh(layer):
    def __init__(self):
        pass
    def eval(self, input):
        return np.tanh(input)
    def inputDerivatives(self, input, outputDerivatives):
        out = self.eval(input)
        return outputDerivatives*(1 - out**2)


class crossEntropy(resultLayer):
    def __init__(self):
        return
    def eval(self, guess, answer):
        return -1* sum(answer * np.log(guess))
    def inputDerivatives(self, guess, answer):
        return answer/guess

class sumSquareError(resultLayer):
    def __init__(self):
        return
    def eval(self, guess, answer):
        return sum((answer - guess)**2)
    def inputDerivatives(self, guess, answer):
        return 2*(answer - guess)

class softMax(layer):
	def __init__(self):
		return
	def eval(self, input):
		num = np.exp(input)
		return num / num.sum()
	def inputDerivatives(self, input, outputDerivatives):
		out = self.eval(input)
		return outputDerivatives*out*(1 - out)
