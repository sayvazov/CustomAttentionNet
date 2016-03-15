import layers
import glimpse
import numpy as np



#Constants
contextKernelSize = 5
contextNumFilters = 10
downsampleHeight = 20
downsampleWidth = 30
r2Length = 10
glimpseSize = 20
glimpseKernelSize = 5
glimpseNumFilters = 10

def AttentionNet(input, labels, inputSize, labelCategories):
    #Context Network layers
    contextConv1 = layers.convLayer(contextKernelSize, contextNumFilters)
    contextConv2 = layers.convLayer(contextKernelSize, contextNumFilters)
    contextConv3 = layers.convLayer(contextKernelSize, contextNumFilters)
    contextSig = layers.sigmoid()
    contextDense = layers.denseLayer(contextNumFilters*downsampleHeight*downsampleWidth, r2Length)

    #Emission layers
    emission = layers.denseLayer(r2Length, 2)

    #Glimpse layers
    glimpseConv1 = layers.convLayer(glimpseKernelSize, glimpseNumFilters)
    glimpseConv2 = layers.convLayer(glimpseKernelSize, glimpseNumFilters)
    glimpseConv3 = layers.convLayer(glimpseKernelSize, glimpseNumFilters)
    glimpseSig = layers.sigmoid()
    glimpseImDense = layers.denseLayer(glimpseNumFilters*glimpseSize*glimpseSize, r2Length)
    glimpseLocDense = layers.denseLayer(2, r2Length)
    glimpseMultiply = layers.elemMulLayer()

    #Recurrent
    r2LSTM = layers.lstmLayer(r2Length, r2Length)
    r1LSTM = layers.lstmLayer(r2Length, r2Length)

    #Classification 
    classDense = layers.denseLayer(r2Length+2, labelCategories)
    classSoftmax = layers.layer()










