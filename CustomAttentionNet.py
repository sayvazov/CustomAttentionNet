import layers
import glimpse
import testing
import numpy as np

#Constants
contextKernelSize = 5
contextNumFilters = 5
downsampleHeight = 21
downsampleWidth = 30
r2Length = 5
glimpseSize = 2
glimpseKernelSize = 5
glimpseNumFilters = 4
labelCategories = 2

#Context Network layers
contextConv1 = layers.convLayer(contextKernelSize, contextNumFilters, 1)
contextConv2 = layers.convLayer(contextKernelSize, contextNumFilters, contextNumFilters)
contextConv3 = layers.convLayer(contextKernelSize, contextNumFilters, contextNumFilters)
contextSig = layers.sigmoid()
contextDense = layers.denseLayer(contextNumFilters*downsampleHeight*downsampleWidth, r2Length)

class AttentionNet(layers.layer):
    def __init__(self):
        
        #Emission layers
        self.emission = layers.denseLayer(r2Length, 2)

        #Glimpse layers
        self.glimpseConv1 = layers.convLayer(glimpseKernelSize, glimpseNumFilters, 1)
        self.glimpseConv2 = layers.convLayer(glimpseKernelSize, glimpseNumFilters,glimpseNumFilters)
        self.glimpseConv3 = layers.convLayer(glimpseKernelSize, glimpseNumFilters, glimpseNumFilters)
        self.glimpseSig = layers.sigmoid()
        self.glimpseImDense = layers.denseLayer(glimpseNumFilters*glimpseSize*glimpseSize, r2Length)
        self.glimpseLocDense = layers.denseLayer(2, r2Length)
        self.glimpseMultiply = layers.elemMulLayer()

        #Recurrent
        self.r2LSTM = layers.lstmLayer(r2Length, r2Length)
        self.r1LSTM = layers.lstmLayer(r2Length, r2Length)
        self.dLstm1Hidden = np.zeros(r2Length)
        self.dLstm2Hidden = np.zeros(r2Length)


        #Classification 
        self.classDense = layers.denseLayer(r2Length+2, labelCategories)
        self.classSoftmax = layers.softMax()
        self.classError = layers.crossEntropy()

        

    def eval(self, glimpse, loc):
        self.gConv1 = self.glimpseConv1.eval(glimpse)
        self.gConv1Sig = self.glimpseSig.eval(self.gConv1)
        self.gConv2 = self.glimpseConv2.eval(self.gConv1Sig)
        self.gConv2Sig = self.glimpseSig.eval(self.gConv2)
        self.gConv3 = self.glimpseConv3.eval(self.gConv2Sig)
        self.gConv3Sig = self.glimpseSig.eval(self.gConv3)
        self.gConv3Reshaped = self.gConv3Sig.flatten()
        self.gImDense = self.glimpseImDense.eval(self.gConv3Reshaped)
        self.gLocDense = self.glimpseLocDense.eval(loc)
        self.gOut = self.glimpseMultiply.eval(self.gImDense, self.gLocDense)
        self.r2 = self.r2LSTM.eval(self.gOut)
        self.r1 = self.r1LSTM.eval(self.gOut)

        self.nextLoc = self.emission.eval(self.r2)
        #print("r1 shape", self.r1.shape)
        #print("Loc shape", len(loc))
        
        self.r1andLoc = np.ma.concatenate( (self.r1, loc))
        self.cDense = self.classDense.eval( self.r1andLoc)
        self.cSoftmax = self.classSoftmax.eval(self.cDense)
        
        return self.nextLoc, self.cSoftmax

    def setUpdate(self, glimpse, loc, nextLocDerivatives, softmaxDerivatives):
        self.dSoftmax = self.classSoftmax.setUpdate(self.cDense, softmaxDerivatives)
        self.dCDense = self.classDense.setUpdate(self.r1andLoc, self.dSoftmax)
        self.dCDenseR1 = self.dCDense[:-2]
        self.dcDenseLoc = self.dCDense[-2:]
        self.dR1, self.dLstm1Hidden = self.r1LSTM.setUpdate(self.gOut, self.dCDenseR1+self.dLstm1Hidden)
        self.dNextLoc = self.emission.setUpdate(self.r2, nextLocDerivatives)
        self.dR2, self.dLstm2Hidden = self.r2LSTM.setUpdate(self.gOut, self.dNextLoc +self.dLstm2Hidden)
        self.dGOutImage, self.dGOutLoc = self.glimpseMultiply.setUpdate(self.gImDense, self.gLocDense, self.dR1 + self.dR2)
        self.dGImDense = self.glimpseImDense.setUpdate(self.gConv3Reshaped, self.dGOutImage)
        self.dGConv3Reshaped = self.dGImDense.reshape(glimpseNumFilters, glimpseSize, glimpseSize)
        self.dGConv3Sig = self.glimpseSig.setUpdate(self.gConv3, self.dGConv3Reshaped)
        self.dGConv3 = self.glimpseConv3.setUpdate(self.gConv2Sig, self.dGConv3Sig)
        self.dGConv2Sig = self.glimpseSig.setUpdate(self.gConv2, self.dGConv3)
        self.dGConv2 = self.glimpseConv2.setUpdate(self.gConv1Sig, self.dGConv2Sig)
        self.dGConv1Sig = self.glimpseSig.setUpdate(self.gConv1, self.dGConv2)
        self.dGConv1 = self.glimpseConv1.setUpdate(glimpse, self.dGConv1Sig)
        return self.dGConv1

    def doUpdate(self, lr):
        #Emission layers
        self.emission.doUpdate(lr)

        #Glimpse layers
        self.glimpseConv1.doUpdate(lr)
        self.glimpseConv2.doUpdate(lr)
        self.glimpseConv3.doUpdate(lr)
        self.glimpseImDense.doUpdate(lr)
        self.glimpseLocDense.doUpdate(lr)

        #Recurrent
        self.r2LSTM.doUpdate(lr)
        self.r1LSTM.doUpdate(lr)


        #Classification 
        self.classDense.doUpdate(lr)

class glimpseNet(layers.layer):
    def eval(self, input, loc):
        return glimpse.takeGlimpse(input, loc[0], loc[1], glimpseSize)
    def setUpdate(self, input, loc, outputDerivatives):
        return [0,0]
    def doUpdate(self):
        return

def test(inputs):
    x1 = inputs[0][0]
    x2 = inputs[0][1]
    y1 = inputs[1][0]
    y2 = inputs[1][1]
    if (2*x1 - 2*x2  + y1 - y2 > 0):
        return 1
    else: 
        return 0

def attentionTest():
    data = np.random.random((30, 2, 2))
    labs = np.array(list(map(test, data)))
    labels = []
    for label in labs:
        if label == 1:
            labels.append([0,1])
        else:
            labels.append([1,0])
    net = AttentionNet()
    glimpser = glimpseNet()
    errorAnalyzer = layers.crossEntropy()
    firstLoc = [0,0]
    for epoch in range(100):
        #print("Started Epcoh ", epoch)
        epochError = 0.0
        for datum, label in zip(data, labels):
            loc = [[0,0]]
            glim = []
            #print("Started forward prop round ")
            for round in range(3):
                #print(round)
                #print("Looking at loc ", loc[-1])
                glim.append(glimpser.eval(datum, loc[-1]))
                nextLoc, soft = net.eval([glim[-1]], loc[-1])
                loc.append(nextLoc)
            error = errorAnalyzer.eval(soft, label)
            epochError += error
            dError = errorAnalyzer.inputDerivatives(soft, label)
            #print(dError)
            dLoc = [[0,0]]
            #print("Started back prop round ")
            for round in range(3):
                #print(round)
                dGlimpse = net.setUpdate([glim[2-round]], loc[2-round], dLoc[-1], dError)
                dLoc.append(glimpser.setUpdate(datum, loc[2-round], dGlimpse))
            net.doUpdate(.001)
        print("Error in epoch ", epoch, " is ",epochError)


attentionTest()










