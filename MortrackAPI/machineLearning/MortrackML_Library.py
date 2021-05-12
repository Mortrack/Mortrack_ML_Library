
"""
   Copyright 2021 Cesar Miranda Meza (alias: Mortrack)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 1 17:15:51 2020
Last updated on Mon May 10 1:38:30 2021

@author: enginer Cesar Miranda Meza (alias: Mortrack)
"""
from ..linearAlgebra import MortrackLinearAlgebraLibrary as mLAL


# IMPORTANT NOTE: Remember to be careful with input argument variables when
#                 you call any method or class because it gets altered in
#                 python ignoring parenting or children logic

"""
DiscreteDistribution()

The Combinations class allows you to get some parameters, through some of its
methods, that describe the dataset characteristics (eg. mean, variance and
standard deviation).
"""   
class DiscreteDistribution:
    
    """
    getMean(samplesList="will contain a matrix of rows and columns, were we want to get the Mean of each rows data point samples")
    
    Returns a matrix (containing only 1 column for all rows within this class
    local variable "samplesList"), were each row will have its corresponding
    mean value.
    
    EXAMPLE CODE:
        matrix_x = [
         [1,2,3],
         [4,5,6],
         [1,5,9]        
        ]
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        dD = mSL.DiscreteDistribution()
        result = dD.getMean(matrix_x)
        
    EXPECTED CODE RESULT:
        result =
        [[2.0], [5.0], [5.0]]
    """
    def getMean(self, samplesList):
        meanList = []
        numberOfRows = len(samplesList)
        numberOfSamplesPerRow = len(samplesList[0])
        for row in range(0, numberOfRows):
            temporalRow = []
            temporalRow.append(0)
            meanList.append(temporalRow)
        for row in range(0, numberOfRows):
            for column in range(0, numberOfSamplesPerRow):
                meanList[row][0] = meanList[row][0] + samplesList[row][column]
            meanList[row][0] = meanList[row][0]/numberOfSamplesPerRow
        return meanList 
    
    """
    getVariance(samplesList="will contain a matrix of rows and columns, were we want to get the Variance of each rows data point samples")
    
    Returns a matrix (containing only 1 column for all rows within this class
    local variable "samplesList"), were each row will have its corresponding
    variance value.
    Remember that Variance is also denoted as the square of sigma
    
    EXAMPLE CODE:
        matrix_x = [
         [1,2,3],
         [4,5,6],
         [1,5,9],
         [1,4,7]
        ]
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        dD = mSL.DiscreteDistribution()
        result = dD.getVariance(matrix_x)
        
    EXPECTED CODE RESULT:
        result =
        [[1.0], [1.0], [16.0], [9.0]]
    """
    def getVariance(self, samplesList):
        numberOfSamplesPerRow = len(samplesList[0])
        if (numberOfSamplesPerRow<2):
            raise Exception('ERROR: The given number of samples must be at least 2.')
        varianceList = []
        numberOfRows = len(samplesList)
        for row in range(0, numberOfRows):
            temporalRow = []
            temporalRow.append(0)
            varianceList.append(temporalRow)
        meanList = self.getMean(samplesList)
        for row in range(0, numberOfRows):
            for column in range(0, numberOfSamplesPerRow):
                varianceList[row][0] = varianceList[row][0] + (samplesList[row][column] - meanList[row][0])**2
            varianceList[row][0] = varianceList[row][0]/(numberOfSamplesPerRow-1)
        return varianceList
    
    """
    getStandardDeviation(samplesList="will contain a matrix of rows and columns, were we want to get the Standard Deviation of each rows data point samples")
    
    Returns a matrix (containing only 1 column for all rows within this class
    local variable "samplesList"), were each row will have its corresponding
    Standard Deviation value.
    Remember that Standard Deviation is also denoted as sigma
    
    EXAMPLE CODE:
        matrix_x = [
         [1,2,3],
         [4,5,6],
         [1,5,9],
         [1,4,7]
        ]
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        dD = mSL.DiscreteDistribution()
        result = dD.getStandardDeviation(matrix_x)
        
    EXPECTED CODE RESULT:
        result =
        [[1.0], [1.0], [16.0], [9.0]]
    """
    def getStandardDeviation(self, samplesList):
        numberOfSamplesPerRow = len(samplesList[0])
        if (numberOfSamplesPerRow<2):
            raise Exception('ERROR: The given number of samples must be at least 2.')
        standardDeviationList = []
        numberOfRows = len(samplesList)
        numberOfSamplesPerRow = len(samplesList[0])
        for row in range(0, numberOfRows):
            temporalRow = []
            temporalRow.append(0)
            standardDeviationList.append(temporalRow)
        meanList = self.getMean(samplesList)
        for row in range(0, numberOfRows):
            for column in range(0, numberOfSamplesPerRow):
                standardDeviationList[row][0] = standardDeviationList[row][0] + (samplesList[row][column] - meanList[row][0])**2
            standardDeviationList[row][0] = (standardDeviationList[row][0]/(numberOfSamplesPerRow-1))**(1/2)
        return standardDeviationList
        
"""
Tdistribution(desiredTrustInterval="Its a float numeric type value that will represent the desired percentage(%) that you desire for your trust interval")

The Combinations class allows you to get some parameters, through some of its
methods, that describe the dataset characteristics (eg. mean, variance and
standard deviation).
"""   
class Tdistribution:
    def __init__(self, desiredTrustInterval):
        self.desiredTrustInterval = desiredTrustInterval
        if ((self.desiredTrustInterval != 95) and (self.desiredTrustInterval != 99) and (self.desiredTrustInterval != 99.9)):
            raise Exception('ERROR: The desired trust interval hasnt been programmed on this class yet.')
    
    """
    getCriticalValue(numberOfSamples="Must have a whole number that represents the number of samples you want to get the critical value from")
    
    Returns a float numeric value which will represent the Critical Value of
    the parameters that you specified (the desired trust interval and the
    number of samples)
    Remember that the T-distribution considers that your data has a normal
    function form tendency.
    
    EXAMPLE CODE:
        matrix_x = [
         [1,2,3],
         [4,5,6],
         [1,5,9],
         [1,4,7]
        ]
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        tD = mSL.Tdistribution(desiredTrustInterval=95)
        result = tD.getCriticalValue(len(matrix_x[0]))
        
    EXPECTED CODE RESULT:
        result =
        4.303
    """
    def getCriticalValue(self, numberOfSamples):
        #  Ecuacion de valores criticos de la distribucion t (la cual supone
        #  que tenemos un dataset con tendencia a la de una funcion normal).
        if ((type(numberOfSamples)!=int) and (numberOfSamples>0)):
            raise Exception('ERROR: The number of samples must have a whole value and not with decimals. Such whole value must be greater than zero.')
        v = numberOfSamples-1
        if (self.desiredTrustInterval == 95):
            if (v==0):
                raise Exception('ERROR: The t distribution mathematical method requires at least 2 samples.')
            if (v<=30):
               v = int(v)
               criticalValue = [12.706,4.303,3.182,2.776,2.571,2.447,2.365,2.306,2.262,2.228,2.201,2.179,2.160,2.145,2.131,2.120,2.110,2.101,2.093,2.086,2.080,2.074,2.069,2.064,2.060,2.056,2.052,2.048,2.045,2.042]             
               return (criticalValue[v-1])
            if ((v>30) and (v<=120)):
                criticalValue = 2.16783333-6.11250000e-03*v+7.26157407e-05*v**2-2.89351852e-07*v**3
                return (criticalValue)
            if ((v>120) and (v<=2400)):
                criticalValue = 1.98105-8.77193e-6*v
                return (criticalValue)
            if (v>2400):
                criticalValue = 1.96
                return (criticalValue)
        if (self.desiredTrustInterval == 99):
            if (v==0):
                raise Exception('ERROR: The t distribution mathematical method requires at least 2 samples.')
            if (v<=30):
               v = int(v)
               criticalValue = [63.656,9.925,5.841,4.604,4.032,3.707,3.499,3.355,3.250,3.196,3.106,3.055,3.012,2.977,2.947,2.921,2.898,2.878,2.861,2.845,2.831,2.819,2.807,2.797,2.787,2.779,2.771,2.763,2.756,2.750]
               return (criticalValue[v-1])
            if ((v>30) and (v<=120)):
                criticalValue = 3.03316667-1.38875000e-02*v+1.68773148e-04*v**2-6.82870370e-07*v**3
                return (criticalValue)
            if ((v>120) and (v<=2400)):
                criticalValue = 2.619-17.982e-6*v
                return (criticalValue)
            if (v>2400):
                criticalValue = 2.576
                return (criticalValue)
        if (self.desiredTrustInterval == 99.9):
            if (v==0):
                raise Exception('ERROR: The t distribution mathematical method requires at least 2 samples.')
            if (v<=30):
               v = int(v)
               criticalValue = [636.578,31.600,12.924,8.610,6.869,5.959,5.408,5.041,4.781,4.587,4.437,4.318,4.221,4.140,4.073,4.015,3.965,3.922,3.883,3.850,3.819,3.792,3.768,3.745,3.725,3.707,3.689,3.674,3.660,3.646]
               return (criticalValue[v-1])
            if ((v>30) and (v<=120)):
                criticalValue = 4.23-2.86250000e-02*v+3.47361111e-04*v**2-1.40277778e-06*v**3
                return (criticalValue)
            if ((v>120) and (v<=2400)):
                criticalValue = 3.37737-36.4035e-6*v
                return (criticalValue)
            if (v>2400):
                criticalValue = 3.290
                return (criticalValue)


class TrustIntervals:
        
    """
    getMeanIntervals(samplesList="Must contain the matrix of the dataset from which you want to get the Mean Intervals",
                     meanList="Must contain the matrix (containing only 1 column for all rows), were each row will have its corresponding mean value.",
                     standardDeviationList="Must contain the matrix (containing only 1 column for all rows), were each row will have its corresponding standard deviation value.",
                     tValue="Must contain a float numeric value that represents the T-Value (Critical Value) required to calculate the mean intervals")
    
    This method returns a matrix with 2 columns:
        * Column 1 = negative mean interval values in the corresponding "n" number of rows
        * Column 2 = positive mean interval values in the corresponding "n" number of rows
    Remember that the T-distribution considers that your data has a normal
    function form tendency.
    
    
    EXAMPLE CODE:
        matrix_x = [
         [1,2,3],
         [4,5,6],
         [1,5,9],
         [1,4,7]
        ]
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        tI = mSL.TrustIntervals()
        dD = mSL.DiscreteDistribution()
        meanList = dD.getMean(matrix_x)
        standardDeviationList = dD.getStandardDeviation(matrix_x)
        tD = mSL.Tdistribution(desiredTrustInterval=95)
        tValue = tD.getCriticalValue(len(matrix_x[0]))
        meanIntervalsList = tI.getMeanIntervals(matrix_x, meanList, standardDeviationList, tValue)
        negativeMeanIntervalList = []
        positiveMeanIntervalList = []
        for row in range(0, len(meanIntervalsList)):
            temporalRow = []
            temporalRow.append(meanIntervalsList[row][0])
            negativeMeanIntervalList.append(temporalRow)
            temporalRow = []
            temporalRow.append(meanIntervalsList[row][1])
            positiveMeanIntervalList.append(temporalRow)
        
        
    EXPECTED CODE RESULT:
        negativeMeanIntervalList =
        [[-0.48433820832295993],
         [2.51566179167704],
         [-4.93735283329184],
         [-3.453014624968879]]
        
        positiveMeanIntervalList =
        [[4.48433820832296],
         [7.48433820832296],
         [14.93735283329184],
         [11.45301462496888]]
    """
    def getMeanIntervals(self, samplesList, meanList, standardDeviationList, tValue):
        # media-(valorT)(s/(n)**(1/2))   <  media  <   media+(valorT)(s/(n)**(1/2))
        meanIntervals = []
        numberOfRows = len(samplesList)
        numberOfSamplesPerRow = len(samplesList[0])
        for row in range(0, numberOfRows):
            temporalRow = []
            temporalRow.append(meanList[row][0] - tValue*standardDeviationList[row][0]/(numberOfSamplesPerRow**(1/2)))
            temporalRow.append(meanList[row][0] + tValue*standardDeviationList[row][0]/(numberOfSamplesPerRow**(1/2)))
            meanIntervals.append(temporalRow)
        return meanIntervals
    
    """
    getPredictionIntervals(samplesList="Must contain the matrix of the dataset from which you want to get the Prediction Intervals",
                     meanList="Must contain the matrix (containing only 1 column for all rows), were each row will have its corresponding mean value.",
                     standardDeviationList="Must contain the matrix (containing only 1 column for all rows), were each row will have its corresponding standard deviation value.",
                     tValue="Must contain a float numeric value that represents the T-Value (Critical Value) required to calculate the Prediction intervals")
    
    This method returns a matrix with 2 columns:
        * Column 1 = negative Prediction interval values in the corresponding "n" number of rows
        * Column 2 = positive Prediction interval values in the corresponding "n" number of rows
    Remember that the T-distribution considers that your data has a normal
    function form tendency.
    
    
    EXAMPLE CODE:
        matrix_x = [
         [1,2,3],
         [4,5,6],
         [1,5,9],
         [1,4,7]
        ]
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        tI = mSL.TrustIntervals()
        dD = mSL.DiscreteDistribution()
        meanList = dD.getMean(matrix_x)
        standardDeviationList = dD.getStandardDeviation(matrix_x)
        tD = mSL.Tdistribution(desiredTrustInterval=95)
        numberOfSamples = len(matrix_x[0])
        tValue = tD.getCriticalValue(numberOfSamples)
        predictionIntervalsList = tI.getPredictionIntervals(numberOfSamples, meanList, standardDeviationList, tValue)
        negativePredictionIntervalList = []
        positivePredictionIntervalList = []
        for row in range(0, len(predictionIntervalsList)):
            temporalRow = []
            temporalRow.append(predictionIntervalsList[row][0])
            negativePredictionIntervalList.append(temporalRow)
            temporalRow = []
            temporalRow.append(predictionIntervalsList[row][1])
            positivePredictionIntervalList.append(temporalRow)
        
        
    EXPECTED CODE RESULT:
        negativePredictionIntervalList =
        [[-2.968676416645919],
         [0.03132358335408103],
         [-14.874705666583676],
         [-10.906029249937756]]
        
        positivePredictionIntervalList =
        [[6.968676416645919],
         [9.96867641664592],
         [24.874705666583676],
         [18.906029249937756]]
    """
    def getPredictionIntervals(self, numberOfSamples, meanList, standardDeviation, tValue):
        # media-(valorT)(s)*(1+1/n)**(1/2)   <  media  <   media+(valorT)(s)*(1+1/n)**(1/2)
        predictionIntervals = []
        numberOfRows = len(meanList)
        for row in range(0, numberOfRows):
            temporalRow = []
            temporalRow.append(meanList[row][0] - tValue*standardDeviation[row][0]*((1+1/numberOfSamples)**(1/2)))
            temporalRow.append(meanList[row][0] + tValue*standardDeviation[row][0]*((1+1/numberOfSamples)**(1/2)))
            predictionIntervals.append(temporalRow)
        return predictionIntervals
    
    
"""
Combinations("The sample list you want to work with")

The Combinations class allows you to get the possible combinations within
the values contained in the "samplesList" variable contained within this class.
"""    
class Combinations:
    def __init__(self, samplesList):
        self.samplesList = samplesList
    
    """
    setSamplesList("The new sample list you want to work with")
    
    This method changes the value of the object's variable "samplesList" to a
    new set of list values that you want to work with through this class
    methods.
    """
    def setSamplesList(self, samplesList):
        self.samplesList = samplesList
        
    """
    getPositionCombinationsList()
    
    Returns all the possible positions of the elements contained within a list
    
    EXAMPLE CODE:
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        combinations = mSL.Combinations([0,1,2])
        result = combinations.getPositionCombinationsList()
        
    EXPECTED CODE RESULT:
        result =
        [[0, 1, 2], [1, 0, 2], [1, 2, 0], [0, 2, 1], [2, 0, 1]]
    """
    def getPositionCombinationsList(self):
        samplesListLength = len(self.samplesList)
        originalSamplesList = self.samplesList
        possibleCombinations = [ [ 0 for i in range(samplesListLength) ] for j in range(samplesListLength**2) ]
        for row in range(0, samplesListLength**2):
            for column in range(0, samplesListLength):
                possibleCombinations[row][column] = originalSamplesList[column]
        possibleCombinationsRow = 0
        for specificDataPoint in range(0, samplesListLength):
            for newDataPointPosition in range(0, samplesListLength):
                possibleCombinations[possibleCombinationsRow].pop(specificDataPoint)
                possibleCombinations[possibleCombinationsRow].insert(newDataPointPosition ,originalSamplesList[specificDataPoint])
                possibleCombinationsRow = possibleCombinationsRow + 1
        possibleCombinationsRow = 0
        while(True):
            for row in range(0, len(possibleCombinations)):
                isRowMatch = True
                for column in range(0, samplesListLength):
                    if (possibleCombinations[possibleCombinationsRow][column] != possibleCombinations[row][column]):
                        isRowMatch = False
                if ((isRowMatch==True) and (possibleCombinationsRow!=row)):
                    possibleCombinations.pop(row)
                    possibleCombinationsRow = possibleCombinationsRow - 1
                    break
            possibleCombinationsRow = possibleCombinationsRow + 1
            if (possibleCombinationsRow==len(possibleCombinations)):
                break
        return possibleCombinations
        
    """
    getCustomizedPermutationList()
    
    Returns a customized form of permutation of the elements contained within a
    list. See code example and expected code result to get a better idea of how
    this method works.
    
    EXAMPLE CODE:
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        combinations = mSL.Combinations([0,1,2])
        result = combinations.getCustomizedPermutationList()
        
    EXPECTED CODE RESULT:
        result =
        [[], [0], [1], [0, 1], [2], [0, 2], [1, 2], [0, 1, 2]]
    """
    def getCustomizedPermutationList(self):
        samplesListLength = len(self.samplesList)
        originalSamplesList = self.samplesList
        customizedPermutations = []
        for row in range(0, 2**samplesListLength):
            temporalRow = []
            for column in range(0, samplesListLength):
                if (((row)&(2**column)) == 2**column):
                    temporalRow.append(originalSamplesList[column])
            customizedPermutations.append(temporalRow)
        return customizedPermutations


"""
DatasetSplitting("x independent variable datapoints to model", "y dependent variable datapoints to model")

The DatasetSplitting Library allows you to split your dataset into training and
test set.
"""    
class DatasetSplitting:
    def __init__(self, x_samplesList, y_samplesList):
        self.y_samplesList = y_samplesList
        self.x_samplesList = x_samplesList
        
    """
    getDatasetSplitted(testSize = "the desired size of the test samples. This value must be greater than zero and lower than one",
                       isSplittingRandom = "True if you want samples to be splitted randomly. False if otherwise is desired")
    
    This method returns a splited dataset into training and test sets.
    
    CODE EXAMPLE1:
        matrix_y = [
                [14.05],
                [10.55],
                [7.55],
                [14.93],
                [9.48],
                [6.59],
                [16.56],
                [13.63],
                [9.23],
                [15.85],
                [11.75],
                [8.78],
                [22.41],
                [18.55],
                [15.93],
                [21.66],
                [17.98],
                [16.44]
                ]
        matrix_x = [
                [75, 15],
                [100, 15],
                [125, 15],
                [75, 17.5],
                [100, 17.5],
                [125, 17.5],
                [75, 20],
                [100, 20],
                [125, 20],
                [75, 22.5],
                [100, 22.5],
                [125, 22.5],
                [75, 25],
                [100, 25],
                [125, 25],
                [75, 27.5],
                [100, 27.5],
                [125, 27.5]
                ]
        
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        dS = mSL.DatasetSplitting(matrix_x, matrix_y)
        datasetSplitResults = dS.getDatasetSplitted(testSize = 0.10, isSplittingRandom = False)
        x_train = datasetSplitResults[0]
        x_test = datasetSplitResults[1]
        y_train = datasetSplitResults[2]
        y_test = datasetSplitResults[3]
        
    EXPECTED CODE1 RESULT:
        x_train = 
        [[125, 15],
         [75, 17.5],
         [100, 17.5],
         [125, 17.5],
         [75, 20],
         [100, 20],
         [125, 20],
         [75, 22.5],
         [100, 22.5],
         [125, 22.5],
         [75, 25],
         [100, 25],
         [125, 25],
         [75, 27.5],
         [100, 27.5],
         [125, 27.5]]
        
        x_test =
        [[75, 15], [100, 15]]
        
        y_train =
        [[7.55],
         [14.93],
         [9.48],
         [6.59],
         [16.56],
         [13.63],
         [9.23],
         [15.85],
         [11.75],
         [8.78],
         [22.41],
         [18.55],
         [15.93],
         [21.66],
         [17.98],
         [16.44]]
        
        y_test =
        [[14.05], [10.55]]
    """
    def getDatasetSplitted(self, testSize = 0.10, isSplittingRandom = True):
        if ((testSize<=0) or (testSize>=1)):
            raise Exception('ERROR: The testSize argument variable must comply the following criteria: 0>testSize<1')
        totalNumberOfSamples = len(self.y_samplesList)
        totalNumberOfColumns = len(self.x_samplesList[0])
        matrix_x = self.x_samplesList
        matrix_y = self.y_samplesList
        x_train = []
        x_test = []
        y_train = []
        y_test = []
        if (isSplittingRandom == True):
            totalNumberOfTestSamples = round(totalNumberOfSamples*testSize)
            for row in range(0, totalNumberOfTestSamples):
                import random
                # random.randrange(start, stop, step)
                nextTestSampleToRetrieve = random.randrange(0,(totalNumberOfSamples-row-1),1)
                temporalRow = []
                for column in range(0, totalNumberOfColumns):
                    temporalRow.append(matrix_x[nextTestSampleToRetrieve][column])
                x_test.append(temporalRow)
                temporalRow = []
                temporalRow.append(matrix_y[nextTestSampleToRetrieve][0])
                y_test.append(temporalRow)
                matrix_x.pop(nextTestSampleToRetrieve)
                matrix_y.pop(nextTestSampleToRetrieve)
            x_train = matrix_x
            y_train = matrix_y
        else:
            totalNumberOfTestSamples = round(totalNumberOfSamples*testSize)
            for row in range(0, totalNumberOfTestSamples):
                temporalRow = []
                for column in range(0, totalNumberOfColumns):
                    temporalRow.append(matrix_x[0][column])
                x_test.append(temporalRow)
                temporalRow = []
                temporalRow.append(matrix_y[0][0])
                y_test.append(temporalRow)
                matrix_x.pop(0)
                matrix_y.pop(0)
            x_train = matrix_x
            y_train = matrix_y
        # We save the current the modeling results
        datasetSplitResults = []
        datasetSplitResults.append(x_train)
        datasetSplitResults.append(x_test)
        datasetSplitResults.append(y_train)
        datasetSplitResults.append(y_test)
        return datasetSplitResults
    
    
"""
FeatureScaling("datapoints you want to apply Feature Scaling to")

The Feature Scaling Library gives several methods to apply feature scaling
techniques to your datasets.
"""    
class FeatureScaling:
    def __init__(self, samplesList):
        self.samplesList = samplesList
        
    """
    getStandarization("preferedMean=prefered Mean",
                      preferedStandardDeviation="prefered Standard Deviation value",
                      isPreferedDataUsed="True to define you will used prefered values. False to define otherwise.")
    
    This method returns a dataset but with the standarization method, of Feature
    Scaling, applied to such dataset. This method will also return the
    calculated mean and the calculated standard deviation value.
    
    CODE EXAMPLE1:
        matrix_x = [
                [75, 15],
                [100, 15],
                [125, 15],
                [75, 17.5],
                [100, 17.5],
                [125, 17.5],
                [75, 20],
                [100, 20],
                [125, 20],
                [75, 22.5],
                [100, 22.5],
                [125, 22.5],
                [75, 25],
                [100, 25],
                [125, 25],
                [75, 27.5],
                [100, 27.5],
                [125, 27.5]
                ]
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        featureScaling = mSL.FeatureScaling(matrix_x)
        normalizedResults = featureScaling.getStandarization()
        preferedMean = normalizedResults[0]
        preferedStandardDeviation = normalizedResults[1]
        normalizedDataPoints = normalizedResults[2]
        
    EXPECTED CODE1 RESULT:
        preferedMean =
        [[100.0, 21.25]]
        
        preferedStandardDeviation =
        [[21.004201260420146, 4.393343895967546]]
        
        normalizedDataPoints =
        [[-1.1902380714238083, -1.422606594884729],
         [0.0, -1.422606594884729],
         [1.1902380714238083, -1.422606594884729],
         [-1.1902380714238083, -0.8535639569308374],
         [0.0, -0.8535639569308374],
         [1.1902380714238083, -0.8535639569308374],
         [-1.1902380714238083, -0.2845213189769458],
         [0.0, -0.2845213189769458],
         [1.1902380714238083, -0.2845213189769458],
         [-1.1902380714238083, 0.2845213189769458],
         [0.0, 0.2845213189769458],
         [1.1902380714238083, 0.2845213189769458],
         [-1.1902380714238083, 0.8535639569308374],
         [0.0, 0.8535639569308374],
         [1.1902380714238083, 0.8535639569308374],
         [-1.1902380714238083, 1.422606594884729],
         [0.0, 1.422606594884729],
         [1.1902380714238083, 1.422606594884729]]
        
 # ------------------------------------------------------------------------- #   
    CODE EXAMPLE2:
        matrix_x = [
                [75, 15],
                [100, 15],
                [125, 15],
                [75, 17.5],
                [100, 17.5],
                [125, 17.5],
                [75, 20],
                [100, 20],
                [125, 20],
                [75, 22.5],
                [100, 22.5],
                [125, 22.5],
                [75, 25],
                [100, 25],
                [125, 25],
                [75, 27.5],
                [100, 27.5],
                [125, 27.5]
                ]
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        featureScaling = mSL.FeatureScaling(matrix_x)
        mean = [[100, 21.25]]
        standardDeviation = [[21.004201260420146, 4.393343895967546]]
        normalizedResults = featureScaling.getStandarization(preferedMean=mean, preferedStandardDeviation=standardDeviation, isPreferedDataUsed = True)
        preferedMean = normalizedResults[0]
        preferedStandardDeviation = normalizedResults[1]
        normalizedDataPoints = normalizedResults[2]
        
    EXPECTED CODE2 RESULT:
        preferedMean =
        [[100.0, 21.25]]
        
        preferedStandardDeviation =
        [[21.004201260420146, 4.393343895967546]]
        
        normalizedDataPoints =
        [[-1.1902380714238083, -1.422606594884729],
         [0.0, -1.422606594884729],
         [1.1902380714238083, -1.422606594884729],
         [-1.1902380714238083, -0.8535639569308374],
         [0.0, -0.8535639569308374],
         [1.1902380714238083, -0.8535639569308374],
         [-1.1902380714238083, -0.2845213189769458],
         [0.0, -0.2845213189769458],
         [1.1902380714238083, -0.2845213189769458],
         [-1.1902380714238083, 0.2845213189769458],
         [0.0, 0.2845213189769458],
         [1.1902380714238083, 0.2845213189769458],
         [-1.1902380714238083, 0.8535639569308374],
         [0.0, 0.8535639569308374],
         [1.1902380714238083, 0.8535639569308374],
         [-1.1902380714238083, 1.422606594884729],
         [0.0, 1.422606594884729],
         [1.1902380714238083, 1.422606594884729]]
    """
    def getStandarization(self, preferedMean=[], preferedStandardDeviation=[], isPreferedDataUsed = False):
        numberOfSamples = len(self.samplesList)
        numberOfColumns = len(self.samplesList[0])
        if (isPreferedDataUsed == True):
            mean = preferedMean
            standardDeviation = preferedStandardDeviation
        else:
            mean = []
            temporalRow = []
            for column in range(0, numberOfColumns):
                temporalRow.append(0)
            mean.append(temporalRow)
            for row in range(0, numberOfSamples):
                for column in range(0, numberOfColumns):
                    mean[0][column] = mean[0][column] + self.samplesList[row][column]
            for column in range(0, numberOfColumns):
                mean[0][column] = mean[0][column]/numberOfSamples
            standardDeviation = []
            temporalRow = []
            for column in range(0, numberOfColumns):
                temporalRow.append(0)
            standardDeviation.append(temporalRow)
            for row in range(0, numberOfSamples):
                for column in range(0, numberOfColumns):
                    standardDeviation[0][column] = standardDeviation[0][column] + (self.samplesList[row][column]-mean[0][column])**2
            for column in range(0, numberOfColumns):
                standardDeviation[0][column] = (standardDeviation[0][column]/(numberOfSamples-1))**(0.5)
        # Now that we have obtained the data we need for the Normalization
        # equation, we now plug in those values in it.
        normalizedDataPoints = []
        for row in range(0, numberOfSamples):
            temporalRow = []
            for column in range(0, numberOfColumns):
                temporalRow.append((self.samplesList[row][column] - mean[0][column])/standardDeviation[0][column])
            normalizedDataPoints.append(temporalRow)
            
        # We save the current the modeling results
        normalizedResults = []
        normalizedResults.append(mean)
        normalizedResults.append(standardDeviation)
        normalizedResults.append(normalizedDataPoints)
        return normalizedResults
    
    """
    getReverseStandarization("preferedMean=prefered Mean",
                      preferedStandardDeviation="prefered Standard Deviation value")
    
    This method returns a dataset but with its original datapoint values before
    having applied the Standarization Feature Scaling method.
    
    CODE EXAMPLE1:
        matrix_x = [
                [-1.1902380714238083, -1.422606594884729],
                [0.0, -1.422606594884729],
                [1.1902380714238083, -1.422606594884729],
                [-1.1902380714238083, -0.8535639569308374],
                [0.0, -0.8535639569308374],
                [1.1902380714238083, -0.8535639569308374],
                [-1.1902380714238083, -0.2845213189769458],
                [0.0, -0.2845213189769458],
                [1.1902380714238083, -0.2845213189769458],
                [-1.1902380714238083, 0.2845213189769458],
                [0.0, 0.2845213189769458],
                [1.1902380714238083, 0.2845213189769458],
                [-1.1902380714238083, 0.8535639569308374],
                [0.0, 0.8535639569308374],
                [1.1902380714238083, 0.8535639569308374],
                [-1.1902380714238083, 1.422606594884729],
                [0.0, 1.422606594884729],
                [1.1902380714238083, 1.422606594884729]
                ]
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        featureScaling = mSL.FeatureScaling(matrix_x)
        mean = [[100, 21.25]]
        standardDeviation = [[21.004201260420146, 4.393343895967546]]
        deNormalizedResults = featureScaling.getReverseStandarization(preferedMean=mean, preferedStandardDeviation=standardDeviation)
        preferedMean = deNormalizedResults[0]
        preferedStandardDeviation = deNormalizedResults[1]
        deNormalizedDataPoints = deNormalizedResults[2]
        
    EXPECTED CODE1 RESULT:
        preferedMean =
        [[100.0, 21.25]]
        
        preferedStandardDeviation =
        [[21.004201260420146, 4.393343895967546]]
        
        deNormalizedDataPoints =
        [[75.0, 15.0],
         [100.0, 15.0],
         [125.0, 15.0],
         [75.0, 17.5],
         [100.0, 17.5],
         [125.0, 17.5],
         [75.0, 20.0],
         [100.0, 20.0],
         [125.0, 20.0],
         [75.0, 22.5],
         [100.0, 22.5],
         [125.0, 22.5],
         [75.0, 25.0],
         [100.0, 25.0],
         [125.0, 25.0],
         [75.0, 27.5],
         [100.0, 27.5],
         [125.0, 27.5]]
    """
    def getReverseStandarization(self, preferedMean, preferedStandardDeviation):
        numberOfSamples = len(self.samplesList)
        numberOfColumns = len(self.samplesList[0])
        
        deNormalizedDataPoints = []
        for row in range(0, numberOfSamples):
            temporalRow = []
            for column in range(0, numberOfColumns):
                temporalRow.append(self.samplesList[row][column]*preferedStandardDeviation[0][column] + preferedMean[0][column])
            deNormalizedDataPoints.append(temporalRow)
            
        # We save the current the modeling results
        deNormalizedResults = []
        deNormalizedResults.append(preferedMean)
        deNormalizedResults.append(preferedStandardDeviation)
        deNormalizedResults.append(deNormalizedDataPoints)
        return deNormalizedResults
        
    """
    setSamplesList(newSamplesList="the new samples list that you wish to work with")
    
    This method sets a new value in the objects local variable "samplesList".
    """
    def setSamplesList(self, newSamplesList):
        self.samplesList = newSamplesList
        
    
"""
The Regression Library gives several different types of coeficients to model
a required data. But notice that the arguments of this class are expected to be
the mean values of both the "x" and the "y" values.

Regression("mean values of the x datapoints to model", "mean values of the y datapoints to model")
"""    
class Regression:
    
    def __init__(self, x_samplesList, y_samplesList):
        self.y_samplesList = y_samplesList
        self.x_samplesList = x_samplesList
    
    def set_xSamplesList(self, x_samplesList):
        self.x_samplesList = x_samplesList
        
    def set_ySamplesList(self, y_samplesList):
        self.y_samplesList = y_samplesList
        
    """
    # ----------------------------------- #
    # ----------------------------------- #
    # ----- STILL UNDER DEVELOPMENT ----- #
    # ----------------------------------- #
    # ----------------------------------- #
    getGaussianRegression()
    
    Returns the best fitting model to predict the behavior of a dataset through
    a Gaussian Regression model that may have any number of independent
    variables (x).
    Note that if no fitting model is found, then this method will swap the
    dependent variables values in such a way that "0"s will be interpretated as
    "1"s and vice-versa to then try again to find at least 1 fitting model to
    your dataset. If this still doenst work, then this method will return
    modeling results will all coefficients with values equal to zero, predicted
    accuracy equal to zero and all predicted values will also equal zero.
    
    CODE EXAMPLE:
        # We will simulate a dataset that you would normally have in its original form
        matrix_x = [
             [2, 3],
             [3, 2],
             [4, 3],
             [3, 4],
             [1, 3],
             [3, 1],
             [5, 3],
             [3, 5]
             ]
        
        matrix_y = [
             [1],
             [1],
             [1],
             [1],
             [0],
             [0],
             [0],
             [0]
             ]
        
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        regression = mSL.Regression(matrix_x, matrix_y)
        modelingResults = regression.getGaussianRegression()
        modelCoefficients = modelingResults[0]
        acurracy = modelingResults[1]
        predictedData = modelingResults[2]
        coefficientDistribution = modelingResults[3]
        allModeledAccuracies = modelingResults[4]
        
    EXPECTED CODE RESULT:
        modelCoefficients =
        [[39.139277579342206],
         [-13.813509557297337],
         [2.302251592882884],
         [-13.813509557296968],
         [2.302251592882836]]
        
        acurracy =
        99.94999999999685
        
        predictedData =
        [[0.9989999999998915],
         [0.9990000000000229],
         [0.9989999999999554],
         [0.9989999999999234],
         [0.0009999999999997621],
         [0.0010000000000001175],
         [0.00099999999999989],
         [0.000999999999999915]]
        # NOTE:"predictedData" will try to give "1" for positive values and "0"
        #       for negative values always, regardless if your negative values
        #       were originally given to the trained model as "-1"s.
        
        coefficientDistribution =
        'Coefficients distribution for the Gaussian function is as follows:    Gaussian = exp(-(bo + b1*x1 + b2*x1^2 + b3*x2 + b4*x2^2 + ... + b_(n-1)*xn + bn*xn^2 ))'
        
        allModeledAccuracies["independent variable distribution used to get a model"]["model accuracy", "model coefficients obtained but with original distribution", "matrix x data"] =
        # NOTE: since this variable contains large amounts of information, it
        #       will not be displayed but only described on how to use it.
    """
    def getGaussianRegression(self):
        from . import MortrackML_Library as mSL
        import math
        numberOfRows = len(self.y_samplesList)
        
        # We re-adapt the current dependent samples (y) so that we can later
        # use them to make the Gaussian function model withouth obtaining 
        # indeterminate values.
        modifiedSamplesList_y = []
        for row in range(0, numberOfRows):
            temporalRow = []
            if ((self.y_samplesList[row][0]!=1) and (self.y_samplesList[row][0]!=-1) and (self.y_samplesList[row][0]!=0) and (self.y_samplesList[row][0]!=0.001) and (self.y_samplesList[row][0]!=0.999)):
                raise Exception('ERROR: One of the dependent (y) data points doesnt have the right format values (eg. 1 or a -1; 1 or a 0; 0.999 or a 0.001).')
            if ((self.y_samplesList[row][0]==1) or (self.y_samplesList[row][0]==0.999)):
                temporalRow.append(0.999)
            if ((self.y_samplesList[row][0]==-1) or (self.y_samplesList[row][0]==0) or self.y_samplesList[row][0]==0.001):
                temporalRow.append(0.001)
            modifiedSamplesList_y.append(temporalRow)
            
        # We modify our current dependent samples (y) to get the dependent
        # samples (y) that we will input to make the Gaussian function model
        modifiedGaussianSamplesList_y = []
        for row in range(0, numberOfRows):
            temporalRow = []
            #temporalRow.append( -math.log(modifiedSamplesList_y[row][0])*2 )
            temporalRow.append( -math.log(modifiedSamplesList_y[row][0]) )
            modifiedGaussianSamplesList_y.append(temporalRow)
        
        # We obtain the independent coefficients of the best fitting model
        # obtained through the Gaussian function (kernel) that we will use to distort
        # the current dimentional spaces that we were originally given by the
        # user
        regression = mSL.Regression(self.x_samplesList, modifiedGaussianSamplesList_y)
        modelingResults = regression.getMultiplePolynomialRegression(orderOfThePolynomial=2, evtfbmip=False)
        allModeledAccuracies = modelingResults[4]
        
        
        # Re-evaluate every obtained model trained through the Multiple
        # Polynomial Regression but this time determining the best fitting
        # model by recalculating each of their accuracies but this time with
        # the right math equation, which would be the gaussian function.
        bestModelingResults = []
        for currentModelingResults in range(0, len(allModeledAccuracies)):
            currentCoefficients = allModeledAccuracies[currentModelingResults][1]
            isComplyingWithGaussCoefficientsSigns = True
            for currentCoefficient in range(0, len(currentCoefficients)):
                if ((currentCoefficients==0) and (currentCoefficients[currentCoefficient][0]<0)):
                    isComplyingWithGaussCoefficientsSigns = False
                else:
                    #if (((currentCoefficient%2)!=0) and (currentCoefficients[currentCoefficient][0]>0)):
                    #    isComplyingWithGaussCoefficientsSigns = False
                    if (((currentCoefficient%2)==0) and (currentCoefficients[currentCoefficient][0]<0)):
                        isComplyingWithGaussCoefficientsSigns = False
            if (isComplyingWithGaussCoefficientsSigns == True):
                # We determine the accuracy of the obtained coefficients
                predictedData = []
                orderOfThePolynomial = 2
                numberOfIndependentVariables = (len(currentCoefficients)-1)
                for row in range(0, numberOfRows):
                    temporalRow = []
                    actualIc = currentCoefficients[0][0]
                    currentOrderOfThePolynomial = 1
                    currentVariable = 0
                    for currentIndependentVariable in range(0, numberOfIndependentVariables):
                        if (currentOrderOfThePolynomial == (orderOfThePolynomial+1)):
                            currentOrderOfThePolynomial = 1
                            currentVariable = currentVariable + 1
                        actualIc = actualIc + currentCoefficients[currentIndependentVariable+1][0]*self.x_samplesList[row][currentVariable]**(currentOrderOfThePolynomial)
                        currentOrderOfThePolynomial = currentOrderOfThePolynomial + 1
                    temporalRow.append(math.exp(-(actualIc)))
                    predictedData.append(temporalRow)
                predictionAcurracy = 0
                numberOfDataPoints = numberOfRows
                for row in range(0, numberOfDataPoints):
                    n2 = modifiedSamplesList_y[row][0]
                    n1 = predictedData[row][0]
                    if ((n1<0.2) and (n2<0.051)):                
                        newAcurracyValueToAdd = 1-n1
                    else:
                        newAcurracyValueToAdd = (1-(abs(n2-n1)/abs(n2)))
                    if (newAcurracyValueToAdd < 0):
                        newAcurracyValueToAdd = 0
                    predictionAcurracy = predictionAcurracy + newAcurracyValueToAdd
                predictionAcurracy = predictionAcurracy/numberOfDataPoints*100
                if (len(bestModelingResults) == 0):
                    # We save the first best fitting modeling result
                    bestModelingResults = []
                    bestModelingResults.append(currentCoefficients)
                    bestModelingResults.append(predictionAcurracy)
                    bestModelingResults.append(predictedData)
                    bestModelingResults.append("Coefficients distribution for the Gaussian function is as follows:    Gaussian = exp(-(bo + b1*x1 + b2*x1^2 + b3*x2 + b4*x2^2 + ... + b_(n-1)*xn + bn*xn^2 ))")
                    allAccuracies = []
                    temporalRow = []
                    temporalRow.append(bestModelingResults[1])
                    temporalRow.append(bestModelingResults[0])
                    temporalRow.append(self.x_samplesList)
                    allAccuracies.append(temporalRow)
                else:
                    if (predictionAcurracy > bestModelingResults[1]):
                        bestModelingResults = []
                        bestModelingResults.append(currentCoefficients)
                        bestModelingResults.append(predictionAcurracy)
                        bestModelingResults.append(predictedData)
                        bestModelingResults.append("Coefficients distribution for the Gaussian function is as follows:    Gaussian = exp(-(bo + b1*x1 + b2*x1^2 + b3*x2 + b4*x2^2 + ... + b_(n-1)*xn + bn*xn^2 ))")
                        temporalRow = []
                        temporalRow.append(predictionAcurracy)
                        temporalRow.append(currentCoefficients)
                        temporalRow.append(self.x_samplesList)
                        allAccuracies.append(temporalRow)
        
        # ------------------------------------------------------------------ #
        # ------------------------------------------------------------------ #
        # if we couldnt obtain a fitting model at all, try again but this time
        # withouth trying to find a perfect gauss form in the resulting model
        # equation
        if (len(bestModelingResults)==0):
            # Re-evaluate every obtained model trained through the Multiple
            # Polynomial Regression but this time determining the best fitting
            # model by recalculating each of their accuracies but this time with
            # the right math equation, which would be the gaussian function.
            bestModelingResults = []
            for currentModelingResults in range(0, len(allModeledAccuracies)):
                currentCoefficients = allModeledAccuracies[currentModelingResults][1]
                # We determine the accuracy of the obtained coefficients
                predictedData = []
                orderOfThePolynomial = 2
                numberOfIndependentVariables = (len(currentCoefficients)-1)
                for row in range(0, numberOfRows):
                    temporalRow = []
                    actualIc = currentCoefficients[0][0]
                    currentOrderOfThePolynomial = 1
                    currentVariable = 0
                    for currentIndependentVariable in range(0, numberOfIndependentVariables):
                        if (currentOrderOfThePolynomial == (orderOfThePolynomial+1)):
                            currentOrderOfThePolynomial = 1
                            currentVariable = currentVariable + 1
                        actualIc = actualIc + currentCoefficients[currentIndependentVariable+1][0]*self.x_samplesList[row][currentVariable]**(currentOrderOfThePolynomial)
                        currentOrderOfThePolynomial = currentOrderOfThePolynomial + 1
                    temporalRow.append(math.exp(-(actualIc)))
                    predictedData.append(temporalRow)
                predictionAcurracy = 0
                numberOfDataPoints = numberOfRows
                for row in range(0, numberOfDataPoints):
                    n2 = modifiedSamplesList_y[row][0]
                    n1 = predictedData[row][0]
                    if ((n1<0.2) and (n2<0.051)):                
                        newAcurracyValueToAdd = 1-n1
                    else:
                        newAcurracyValueToAdd = (1-(abs(n2-n1)/abs(n2)))
                    if (newAcurracyValueToAdd < 0):
                        newAcurracyValueToAdd = 0
                    predictionAcurracy = predictionAcurracy + newAcurracyValueToAdd
                predictionAcurracy = predictionAcurracy/numberOfDataPoints*100
                if (len(bestModelingResults) == 0):
                    # We save the first best fitting modeling result
                    bestModelingResults = []
                    bestModelingResults.append(currentCoefficients)
                    bestModelingResults.append(predictionAcurracy)
                    bestModelingResults.append(predictedData)
                    bestModelingResults.append("Coefficients distribution for the Gaussian function is as follows:    Gaussian = exp(-(bo + b1*x1 + b2*x1^2 + b3*x2 + b4*x2^2 + ... + b_(n-1)*xn + bn*xn^2 ))")
                    allAccuracies = []
                    temporalRow = []
                    temporalRow.append(bestModelingResults[1])
                    temporalRow.append(bestModelingResults[0])
                    temporalRow.append(self.x_samplesList)
                    allAccuracies.append(temporalRow)
                else:
                    if (predictionAcurracy > bestModelingResults[1]):
                        bestModelingResults = []
                        bestModelingResults.append(currentCoefficients)
                        bestModelingResults.append(predictionAcurracy)
                        bestModelingResults.append(predictedData)
                        bestModelingResults.append("Coefficients distribution for the Gaussian function is as follows:    Gaussian = exp(-(bo + b1*x1 + b2*x1^2 + b3*x2 + b4*x2^2 + ... + b_(n-1)*xn + bn*xn^2 ))")
                        temporalRow = []
                        temporalRow.append(predictionAcurracy)
                        temporalRow.append(currentCoefficients)
                        temporalRow.append(self.x_samplesList)
                        allAccuracies.append(temporalRow)
                        
        if (len(bestModelingResults)==0):
            # We save the first best fitting modeling result
            bestModelingResults = []
            temporalRow = []
            currentCoefficients = []
            for row in range(0, len(allModeledAccuracies[0][1])):
                temporalRow.append(0)
                currentCoefficients.append(temporalRow)
            temporalRow = []
            predictedData = []
            for row in range(0, numberOfRows):
                temporalRow.append(0)
                predictedData.append(temporalRow)
            bestModelingResults.append(currentCoefficients)
            bestModelingResults.append(0)
            bestModelingResults.append(predictedData)
            bestModelingResults.append("Coefficients distribution for the Gaussian function is as follows:    Gaussian = exp(-(bo + b1*x1 + b2*x1^2 + b3*x2 + b4*x2^2 + ... + b_(n-1)*xn + bn*xn^2 ))")
            allAccuracies = []
            temporalRow = []
            temporalRow.append(bestModelingResults[1])
            temporalRow.append(bestModelingResults[0])
            temporalRow.append(self.x_samplesList)
            allAccuracies.append(temporalRow)
        # We include all the reports of all the models studied to the reporting
        # variable that contains the report of the best fitting model and we
        # then return it
        bestModelingResults.append(allAccuracies)
        return bestModelingResults
    
    """
    getLinearLogisticRegression(evtfbmip="True to indicate to Eliminate Variables To Find Better Model If Possible. False if the contrary is desired")
    
    This method returns the best fitting Logistic Regression model to be able
    to predict a classification problem that can have any number of
    independent variables (x).
    
    CODE EXAMPLE:
        matrix_x = [
                [0,2],
                [1,3],
                [2,4],
                [3,5],
                [4,6],
                [5,7],
                [6,8],
                [7,9],
                [8,10],
                [9,11]
                ]
        
        matrix_y = [
                [0],
                [0],
                [1],
                [0],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1]
                ]
        
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        regression = mSL.Regression(matrix_x, matrix_y)
        # evtfbmip stands for "Eliminate Variables To Find Better Model If Possible"
        modelingResults = regression.getLinearLogisticRegression(evtfbmip=True)
        modelCoefficients = modelingResults[0]
        acurracy = modelingResults[1]
        predictedData = modelingResults[2]
        coefficientDistribution = modelingResults[3]
        allModeledAccuracies = modelingResults[4]
        
    EXPECTED CODE RESULT:
        modelCoefficients =
        [[4.395207586412653], [5.985854141495452], [-4.395207586412653]]
        
        acurracy =
        80.02122762886552
        
        predictedData =
        [[0.012185988957723588],
         [0.05707820342364075],
         [0.22900916243958236],
         [0.5930846789223594],
         [0.8773292738274195],
         [0.9722944298625625],
         [0.9942264149220237],
         [0.9988179452639562],
         [0.9997588776328182],
         [0.9999508513195541]]
        
        coefficientDistribution =
        'Coefficients distribution is as follows: p = (exp(bo + b1*x1 + b2*x2 + ... + bn*xn))/(1 + exp(bo + b1*x1 + b2*x2 + ... + bn*xn))'
        
        allModeledAccuracies["independent variable distribution used to get a model"]["model accuracy", "model coefficients obtained but with original distribution", "matrix x data"] =
        # NOTE: since this variable contains large amounts of information, it
        #       will not be displayed but only described on how to use it.
    """
    def getLinearLogisticRegression(self, evtfbmip=True):
        from . import MortrackML_Library as mSL
        import math
        getOptimizedRegression = evtfbmip
        numberOfRows = len(self.y_samplesList)
        matrix_x = self.x_samplesList
        modifiedSamplesList_y = []
        for row in range(0, numberOfRows):
            temporalRow = []
            if ((self.y_samplesList[row][0]!=1) and (self.y_samplesList[row][0]!=0)):
                raise Exception('ERROR: One of the dependent (y) data points doesnt have a 1 or a 0 as value.')
            if (self.y_samplesList[row][0] == 1):
                temporalRow.append(0.999)
            if (self.y_samplesList[row][0] == 0):
                temporalRow.append(0.001)
            modifiedSamplesList_y.append(temporalRow)
        matrix_y = []
        for row in range(0, numberOfRows):
            temporalRow = []
            temporalRow.append(math.log(modifiedSamplesList_y[row][0]/(1-modifiedSamplesList_y[row][0])))
            matrix_y.append(temporalRow)
        regression = mSL.Regression(matrix_x, matrix_y)
        modelingResults = regression.getMultipleLinearRegression(evtfbmip = getOptimizedRegression)
        coefficients = modelingResults[0]

        # We determine the accuracy of the obtained coefficientsfor the 
        # Probability Equation of the Logistic Regression Equation
        predictedData = []
        numberOfIndependentVariables = len(matrix_x[0])
        for row in range(0, len(matrix_y)):
            temporalRow = []
            actualIc = coefficients[0][0]
            for currentIndependentVariable in range(0, numberOfIndependentVariables):
                actualIc = actualIc + coefficients[currentIndependentVariable+1][0]*matrix_x[row][currentIndependentVariable]
            actualIc = math.exp(actualIc)
            actualIc = actualIc/(1+actualIc)
            temporalRow.append(actualIc)
            predictedData.append(temporalRow)
        
        predictionAcurracy = 0
        numberOfDataPoints = numberOfRows
        for row in range(0, numberOfDataPoints):
            n2 = self.y_samplesList[row][0]
            n1 = predictedData[row][0]
            if (n2 == 0):
                n2 = 0.001
                if (n1 < 0.2):                
                    newAcurracyValueToAdd = 1-n1
                else:
                    newAcurracyValueToAdd = (1-(abs(n2-n1)/abs(n2)))
            else:
                newAcurracyValueToAdd = (1-(abs(n2-n1)/abs(n2)))
            if (newAcurracyValueToAdd < 0):
                newAcurracyValueToAdd = 0
            predictionAcurracy = predictionAcurracy + newAcurracyValueToAdd
        predictionAcurracy = predictionAcurracy/numberOfDataPoints*100        
        
        # We save the current the modeling results
        bestModelingResults = []
        bestModelingResults.append(coefficients)
        bestModelingResults.append(predictionAcurracy)
        bestModelingResults.append(predictedData)
        bestModelingResults.append("Coefficients distribution is as follows: p = (exp(bo + b1*x1 + b2*x2 + ... + bn*xn))/(1 + exp(bo + b1*x1 + b2*x2 + ... + bn*xn))")
        allAccuracies = []
        temporalRow = []
        temporalRow.append(bestModelingResults[1])
        temporalRow.append(bestModelingResults[0])
        temporalRow.append(self.x_samplesList)
        allAccuracies.append(temporalRow)
        bestModelingResults.append(allAccuracies)
        return bestModelingResults
    
    """
    getLinearRegression(isClassification="set to True if you are solving a classification problem. False if otherwise")
    
    Returns the best fitting model to predict the behavior of a dataset through
    a regular Linear Regression model. Note that this method can only solve
    regression problems that have 1 independent variable (x).
    
    CODE EXAMPLE:
        matrix_x = [
                [0],
                [1],
                [2],
                [3],
                [4],
                [5],
                [6],
                [7],
                [8],
                [9]
                ]
        matrix_y = [
                [8.5],
                [9.7],
                [10.7],
                [11.5],
                [12.1],
                [14],
                [13.3],
                [16.2],
                [17.3],
                [17.5]
                ]
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        regression = mSL.Regression(matrix_x, matrix_y)
        modelingResults = regression.getLinearRegression(isClassification=False)
        modelCoefficients = modelingResults[0]
        acurracy = modelingResults[1]
        predictedData = modelingResults[2]
        coefficientDistribution = modelingResults[3]
        allModeledAccuracies = modelingResults[4]
        
    EXPECTED CODE RESULT:
        modelCoefficients =
        [[8.470909090909096], [1.0242424242424237]]
        
        acurracy =
        97.05959379759686
        
        predictedData =
        [[8.470909090909096],
         [9.49515151515152],
         [10.519393939393943],
         [11.543636363636367],
         [12.56787878787879],
         [13.592121212121214],
         [14.616363636363639],
         [15.640606060606062],
         [16.664848484848484],
         [17.689090909090908]]
        
        coefficientDistribution =
        'Coefficients distribution is as follows: y = b + m*x'
        
        allModeledAccuracies["independent variable distribution used to get a model"]["model accuracy", "model coefficients obtained but with original distribution", "matrix x data"] =
        # NOTE: since this variable contains large amounts of information, it
        #       will not be displayed but only described on how to use it.
    """
    def getLinearRegression(self, isClassification=True):
        from ..linearAlgebra import MortrackLinearAlgebraLibrary as mLAL
        matrixMath = mLAL.MatrixMath()
        x_samples = matrixMath.getTransposedMatrix(self.x_samplesList)
        y_samples = matrixMath.getTransposedMatrix(self.y_samplesList)
        x_length = len(x_samples[0])
        y_length = len(y_samples[0])
        if x_length != y_length:
            raise Exception('Dependent Variable has a different vector size than Independent Variable')
        x_mean = 0
        x_squared_mean = 0
        y_mean = 0
        xy_mean = 0
        for n in range (0, x_length):
            x_mean += x_samples[0][n]
            x_squared_mean += x_samples[0][n]*x_samples[0][n]
            y_mean += y_samples[0][n]
            xy_mean += x_samples[0][n]*y_samples[0][n]
        x_mean = x_mean/x_length
        x_squared_mean = x_squared_mean/x_length
        y_mean = y_mean/y_length
        xy_mean = xy_mean/x_length
        m = ( (x_mean*y_mean - xy_mean) / (x_mean**2 - x_squared_mean) )
        # m = ( (mean(xs)*mean(ys) - mean(xs*ys)) / (mean(xs)*mean(xs) - mean(xs*xs)) )
        b = y_mean - m*x_mean
        matrix_b = [[b], [m]]
        
        # We determine the accuracy of the obtained coefficients
        predictedData = []
        bothMatrixRowLength = len(self.y_samplesList)
        for row in range(0, bothMatrixRowLength):
            temporalRow = []
            actualIc = matrix_b[0][0] + matrix_b[1][0]*self.x_samplesList[row][0]
            temporalRow.append(actualIc)
            predictedData.append(temporalRow)
        predictionAcurracy = 0
        numberOfDataPoints = bothMatrixRowLength
        for row in range(0, numberOfDataPoints):
            n2 = self.y_samplesList[row][0]
            n1 = predictedData[row][0]
            if (isClassification == False):
                if (((n1*n2) != 0)):
                    newAcurracyValueToAdd = (1-(abs(n2-n1)/abs(n2)))
                    if (newAcurracyValueToAdd < 0):
                        newAcurracyValueToAdd = 0
                    predictionAcurracy = predictionAcurracy + newAcurracyValueToAdd
            if (isClassification == True):
                if (abs(n1) > abs(n2)): # n2 has to be the one with the highest value with respect to n1
                    n2 = predictedData[row][0]
                    n1 = self.y_samplesList[row][0]
                if ((n1==0) and (n2>=-1 and n2<=1) and (n2!=0)):
                    predictionAcurracy = predictionAcurracy + ((1-abs(n2))/(1-n1))
                if (n1==n2):
                    predictionAcurracy = predictionAcurracy + 1
        predictionAcurracy = predictionAcurracy/numberOfDataPoints*100
        
        # We save the current the modeling results
        bestModelingResults = []
        bestModelingResults.append(matrix_b)
        bestModelingResults.append(predictionAcurracy)
        bestModelingResults.append(predictedData)
        bestModelingResults.append("Coefficients distribution is as follows: y = b + m*x")
        allAccuracies = []
        temporalRow = []
        temporalRow.append(bestModelingResults[1])
        temporalRow.append(bestModelingResults[0])
        temporalRow.append(self.x_samplesList)
        allAccuracies.append(temporalRow)
        bestModelingResults.append(allAccuracies)
        return bestModelingResults
        
    """
    getMultipleLinearRegression(evtfbmip="True to indicate to Eliminate Variables To Find Better Model If Possible. False if the contrary is desired",
                                isClassification="set to True if you are solving a classification problem. False if otherwise")
    
    Returns the best fitting model of a regression problem that has any number
    of independent variables (x) through the Multiple Linear Regression method.
    
    EXAMPLE CODE:
        # matrix_y = [expectedResult]
        matrix_y = [
                [25.5],
                [31.2],
                [25.9],
                [38.4],
                [18.4],
                [26.7],
                [26.4],
                [25.9],
                [32],
                [25.2],
                [39.7],
                [35.7],
                [26.5]
                ]
        # matrix_x = [variable1, variable2, variable3]
        matrix_x = [
                [1.74, 5.3, 10.8],
                [6.32, 5.42, 9.4],
                [6.22, 8.41, 7.2],
                [10.52, 4.63, 8.5],
                [1.19, 11.6, 9.4],
                [1.22, 5.85, 9.9],
                [4.1, 6.62, 8],
                [6.32, 8.72, 9.1],
                [4.08, 4.42, 8.7],
                [4.15, 7.6, 9.2],
                [10.15, 4.83, 9.4],
                [1.72, 3.12, 7.6],
                [1.7, 5.3, 8.2]
                ]
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        regression = mSL.Regression(matrix_x, matrix_y)
        # "evtfbmip" stands for "Eliminate Variables To Find Better Model If Possible"
        modelingResults = regression.getMultipleLinearRegression(evtfbmip = True, isClassification=False)
        modelCoefficients = modelingResults[0]
        acurracy = modelingResults[1]
        predictedData = modelingResults[2]
        coefficientDistribution = modelingResults[3]
        allModeledAccuracies = modelingResults[4]
        
    RESULT OF CODE:
        modelCoefficients =
        [[36.094678333151364], [1.030512601856226], [-1.8696429022156238], [0]]
        
        acurracy =
        94.91286851439088
        
        predictedData =
        [[27.97866287863839],
         [32.47405344687403],
         [26.780769909063693],
         [38.27922426742052],
         [15.633130663659042],
         [26.414492729454558],
         [27.942743988094456],
         [26.30423186956247],
         [32.03534812093171],
         [26.162019574015964],
         [37.5240060242906],
         [32.03387415343133],
         [27.937442374564142]]
        
        coefficientDistribution =
        'Coefficients distribution is as follows: y = bo + b1*x1 + b2*x2 + b3*x3 + ... + bn*xn'
        
        allModeledAccuracies["independent variable distribution used to get a model"]["model accuracy", "model coefficients obtained but with original distribution", "matrix x data"] =
        # NOTE: since this variable contains large amounts of information, it
        #       will not be displayed but only described on how to use it.
    """
    def getMultipleLinearRegression(self, evtfbmip = False, isClassification=True):
        # We import the libraries we want to use and we create the class we
        # use from it
        from ..linearAlgebra import MortrackLinearAlgebraLibrary as mLAL
        matrixMath = mLAL.MatrixMath()
        # We define the variables to use within our code
        matrix_x = self.x_samplesList
        matrix_y = self.y_samplesList
        rowLengthOfBothMatrixes = len(matrix_y)
        numberOfIndependentVariables = len(matrix_x[0])
        # ----- WE GET THE FIRST MODEL EVALUATION RESULTS ----- #
        # MATRIX X MATHEMATICAL PROCEDURE to create a matrix that contains
        # the x values of the following equation we want to solve and in that
        # same variable formation:
        # y = bo + b1*x1 + b2*x2 + b3*x3 + ... + bn*xn
        currentMatrix_x = []
        for row in range(0, rowLengthOfBothMatrixes):
            temporalRow = []
            temporalRow.append(1)
            for currentIndependentVariable in range(0, numberOfIndependentVariables):
                temporalRow.append(matrix_x[row][currentIndependentVariable])
            currentMatrix_x.append(temporalRow)
        originalMatrix_x = currentMatrix_x
            
        # We get the Transposed matrix of matrix X. NOTE: We create a temporal
        # variable to save matrix x because remember that in python, children
        # and parent inheritance is ignored when using clases
        temporalMatrix1 = currentMatrix_x
        transposedMatrix_X = matrixMath.getTransposedMatrix(temporalMatrix1)
        # WE GET MATRIX A.  NOTE: We create a temporal
        # variable to save matrix x because remember that in python, children
        # and parent inheritance is ignored when using clases
        temporalMatrix1 = transposedMatrix_X
        temporalMatrix2 = currentMatrix_x
        matrix_A = matrixMath.getMultiplication(temporalMatrix1, temporalMatrix2)
        # WE GET MATRIX g. NOTE: We create a temporal
        # variable to save matrix x because remember that in python, children
        # and parent inheritance is ignored when using clases
        temporalMatrix1 = transposedMatrix_X
        temporalMatrix2 = matrix_y
        matrix_g = matrixMath.getMultiplication(temporalMatrix1, temporalMatrix2)
        # We get inverse matrix of matrix A.
        inversedMatrix_A = matrixMath.getInverse(matrix_A)
        # We get matrix b, which will contain the coeficient values
        matrix_b = matrixMath.getMultiplication(inversedMatrix_A, matrix_g)
        
        # We determine the accuracy of the obtained coefficients
        predictedData = []
        for row in range(0, len(matrix_y)):
            temporalRow = []
            actualIc = matrix_b[0][0]
            for currentIndependentVariable in range(0, numberOfIndependentVariables):
                actualIc = actualIc + matrix_b[currentIndependentVariable+1][0]*matrix_x[row][currentIndependentVariable]
            temporalRow.append(actualIc)
            predictedData.append(temporalRow)
        predictionAcurracy = 0
        numberOfDataPoints = len(matrix_y)
        for row in range(0, numberOfDataPoints):
            n2 = matrix_y[row][0]
            n1 = predictedData[row][0]
            if (isClassification == False):
                if (((n1*n2) != 0)):
                    newAcurracyValueToAdd = (1-(abs(n2-n1)/abs(n2)))
                    if (newAcurracyValueToAdd < 0):
                        newAcurracyValueToAdd = 0
                    predictionAcurracy = predictionAcurracy + newAcurracyValueToAdd
            if (isClassification == True):
                if (abs(n1) > abs(n2)): # n2 has to be the one with the highest value with respect to n1
                    n2 = predictedData[row][0]
                    n1 = matrix_y[row][0]
                if ((n1==0) and (n2>=-1 and n2<=1) and (n2!=0)):
                    predictionAcurracy = predictionAcurracy + ((1-abs(n2))/(1-n1))
                if (n1==n2):
                    predictionAcurracy = predictionAcurracy + 1
        predictionAcurracy = predictionAcurracy/numberOfDataPoints*100
        
        # We save the current the modeling results
        bestModelingResults = []
        bestModelingResults.append(matrix_b)
        bestModelingResults.append(predictionAcurracy)
        bestModelingResults.append(predictedData)
        bestModelingResults.append("Coefficients distribution is as follows: y = bo + b1*x1 + b2*x2 + b3*x3 + ... + bn*xn")
        allAccuracies = []
        temporalRow = []
        temporalRow.append(bestModelingResults[1])
        temporalRow.append(bestModelingResults[0])
        temporalRow.append(originalMatrix_x)
        allAccuracies.append(temporalRow)
        
        # ----- WE START SEARCHING FOR THE BEST MODELING RESULTS ----- #
        # We define a variable to save the search patterns in original matrix x
        from .MortrackML_Library import Combinations
        possibleCombinations = []
        for n in range (0, len(originalMatrix_x[0])):
            possibleCombinations.append(n)
        combinations = Combinations(possibleCombinations)
        searchPatterns = combinations.getPositionCombinationsList()
        searchPatterns.pop(0) # We remove the first one because we already did it
        # We start to search for the coefficients that give us the best accuracy
        for currentSearchPattern in range(0, len(searchPatterns)):
            currentMatrix_x = [ [ 0 for i in range(len(originalMatrix_x[0])) ] for j in range(rowLengthOfBothMatrixes) ]
            # We assign the current distribution that we want to study of the
            # variables of the matrix x, to evaluate its resulting regression
            # coefficients
            for currentColumnOfMatrix_x in range(0, len(originalMatrix_x[0])):
                for column in range(0, len(originalMatrix_x[0])):
                    if (searchPatterns[currentSearchPattern][currentColumnOfMatrix_x] == column):
                        for row in range(0, rowLengthOfBothMatrixes):
                            currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][column]
            # We get the Transposed matrix of matrix X. NOTE: We create a temporal
            # variable to save matrix x because remember that in python, children
            # and parent inheritance is ignored when using clases
            temporalMatrix1 = currentMatrix_x
            transposedMatrix_X = matrixMath.getTransposedMatrix(temporalMatrix1)
            # WE GET MATRIX A.  NOTE: We create a temporal
            # variable to save matrix x because remember that in python, children
            # and parent inheritance is ignored when using clases
            temporalMatrix1 = transposedMatrix_X
            temporalMatrix2 = currentMatrix_x
            matrix_A = matrixMath.getMultiplication(temporalMatrix1, temporalMatrix2)
            # WE GET MATRIX g. NOTE: We create a temporal
            # variable to save matrix x because remember that in python, children
            # and parent inheritance is ignored when using clases
            temporalMatrix1 = transposedMatrix_X
            temporalMatrix2 = matrix_y
            matrix_g = matrixMath.getMultiplication(temporalMatrix1, temporalMatrix2)
            # We get inverse matrix of matrix A.
            inversedMatrix_A = matrixMath.getInverse(matrix_A)
            # We get matrix b, which will contain the coeficient values
            matrix_b = matrixMath.getMultiplication(inversedMatrix_A, matrix_g)
            
            # ----- WE DETERMINE THE ACCURACY OF THE OBTAINED COEFFICIENTS ----- #
            # We re-arrange the obtained coefficients to then evaluate this
            # model
            currentMatrix_b = [ [ 0 for i in range(1) ] for j in range(len(originalMatrix_x[0])) ]
            for row in range(0, len(originalMatrix_x[0])):
                trueRowOfCoefficient = searchPatterns[currentSearchPattern][row]
                currentMatrix_b[trueRowOfCoefficient][0] = matrix_b[row][0]
            # We obtain the predicted data through the current obtained
            # coefficients
            predictedData = []
            for row in range(0, len(matrix_y)):
                temporalRow = []
                actualIc = currentMatrix_b[0][0]
                for currentIndependentVariable in range(0, numberOfIndependentVariables):
                    actualIc = actualIc + currentMatrix_b[currentIndependentVariable+1][0]*self.x_samplesList[row][currentIndependentVariable]
                temporalRow.append(actualIc)
                predictedData.append(temporalRow)
            
            predictionAcurracy = 0
            numberOfDataPoints = len(matrix_y)            
            for row in range(0, numberOfDataPoints):
                n2 = matrix_y[row][0]
                n1 = predictedData[row][0]
                if (isClassification == False):
                    if (((n1*n2) != 0)):
                        newAcurracyValueToAdd = (1-(abs(n2-n1)/abs(n2)))
                        if (newAcurracyValueToAdd < 0):
                            newAcurracyValueToAdd = 0
                        predictionAcurracy = predictionAcurracy + newAcurracyValueToAdd
                if (isClassification == True):
                    if (abs(n1) > abs(n2)): # n2 has to be the one with the highest value with respect to n1
                        n2 = predictedData[row][0]
                        n1 = matrix_y[row][0]
                    if ((n1==0) and (n2>=-1 and n2<=1) and (n2!=0)):
                        predictionAcurracy = predictionAcurracy + ((1-abs(n2))/(1-n1))
                    if (n1==n2):
                        predictionAcurracy = predictionAcurracy + 1
            predictionAcurracy = predictionAcurracy/numberOfDataPoints*100
            temporalRow = []
            temporalRow.append(predictionAcurracy)
            temporalRow.append(currentMatrix_b)
            temporalRow.append(currentMatrix_x)
            allAccuracies.append(temporalRow)
            
            # We save the current the modeling results if they are better than
            # the actual best
            currentBestAccuracy = bestModelingResults[1]
            if (predictionAcurracy > currentBestAccuracy):
                bestModelingResults = []
                bestModelingResults.append(currentMatrix_b)
                bestModelingResults.append(predictionAcurracy)
                bestModelingResults.append(predictedData)
                bestModelingResults.append("Coefficients distribution is as follows: y = bo + b1*x1 + b2*x2 + b3*x3 + ... + bn*xn")
        if (evtfbmip == True):
            # ----------------------------------------------------------------------------------------------- #
            # ----- We now get all possible combinations/permutations with the elements of our equation ----- #
            # ----------------------------------------------------------------------------------------------- #
            customizedPermutations = combinations.getCustomizedPermutationList()
            customizedPermutations.pop(0) # We remove the null value
            customizedPermutations.pop(len(customizedPermutations)-1) # We remove the last one because we already did it
            for actualPermutation in range(0, len(customizedPermutations)):
                newOriginalMatrix_x = []
                for row in range(0, rowLengthOfBothMatrixes):
                    temporalRow = []
                    for column in range(0, len(customizedPermutations[actualPermutation])):
                        temporalRow.append(originalMatrix_x[row][customizedPermutations[actualPermutation][column]])
                    newOriginalMatrix_x.append(temporalRow)
                
            # ----- WE START SEARCHING FOR THE BEST MODELING RESULTS USING CURRENT PERMUTATION ----- #
                # We define a variable to save the search patterns in original matrix x
                possibleCombinations = []
                for n in range (0, len(newOriginalMatrix_x[0])):
                    possibleCombinations.append(n)
                combinations = Combinations(possibleCombinations)
                searchPatterns = combinations.getPositionCombinationsList()
                
                # We start to search for the coefficients that give us the best accuracy
                for currentSearchPattern in range(0, len(searchPatterns)):
                    currentMatrix_x = [ [ 0 for i in range(len(newOriginalMatrix_x[0])) ] for j in range(rowLengthOfBothMatrixes) ]
                    # We assign the current distribution that we want to study of the
                    # variables of the matrix x, to evaluate its resulting regression
                    # coefficients
                    for currentColumnOfMatrix_x in range(0, len(newOriginalMatrix_x[0])):
                        for column in range(0, len(newOriginalMatrix_x[0])):
                            if (customizedPermutations[actualPermutation][searchPatterns[currentSearchPattern][currentColumnOfMatrix_x]] == column):
                                for row in range(0, rowLengthOfBothMatrixes):
                                    currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][column]
                    # We get the Transposed matrix of matrix X. NOTE: We create a temporal
                    # variable to save matrix x because remember that in python, children
                    # and parent inheritance is ignored when using clases
                    temporalMatrix1 = currentMatrix_x
                    transposedMatrix_X = matrixMath.getTransposedMatrix(temporalMatrix1)
                    # WE GET MATRIX A.  NOTE: We create a temporal
                    # variable to save matrix x because remember that in python, children
                    # and parent inheritance is ignored when using clases
                    temporalMatrix1 = transposedMatrix_X
                    temporalMatrix2 = currentMatrix_x
                    matrix_A = matrixMath.getMultiplication(temporalMatrix1, temporalMatrix2)
                    # WE GET MATRIX g. NOTE: We create a temporal
                    # variable to save matrix x because remember that in python, children
                    # and parent inheritance is ignored when using clases
                    temporalMatrix1 = transposedMatrix_X
                    temporalMatrix2 = matrix_y
                    matrix_g = matrixMath.getMultiplication(temporalMatrix1, temporalMatrix2)
                    # We get inverse matrix of matrix A.
                    inversedMatrix_A = matrixMath.getInverse(matrix_A)
                    # We get matrix b, which will contain the coeficient values
                    matrix_b = matrixMath.getMultiplication(inversedMatrix_A, matrix_g)
                    
                    # ----- WE DETERMINE THE ACCURACY OF THE OBTAINED COEFFICIENTS ----- #
                    # We re-arrange the obtained coefficients to then evaluate this
                    # model
                    currentMatrix_b = [ [ 0 for i in range(1) ] for j in range(len(originalMatrix_x[0])) ]
                    for row in range(0, len(newOriginalMatrix_x[0])):
                        trueRowOfCoefficient = customizedPermutations[actualPermutation][searchPatterns[currentSearchPattern][row]]
                        currentMatrix_b[trueRowOfCoefficient][0] = matrix_b[row][0]
                    # We obtain the predicted data through the current obtained
                    # coefficients
                    newNumberOfIndependentVariables = len(currentMatrix_x[0])
                    predictedData = []
                    for row in range(0, len(matrix_y)):
                        temporalRow = []
                        actualIc = currentMatrix_b[0][0]
                        for currentIndependentVariable in range(0, (newNumberOfIndependentVariables-1)):
                            actualIc = actualIc + currentMatrix_b[currentIndependentVariable+1][0]*self.x_samplesList[row][currentIndependentVariable]
                        temporalRow.append(actualIc)
                        predictedData.append(temporalRow)
                    
                    predictionAcurracy = 0
                    numberOfDataPoints = len(matrix_y)
                    for row in range(0, numberOfDataPoints):
                        n2 = matrix_y[row][0]
                        n1 = predictedData[row][0]
                        if (isClassification == False):
                            if (((n1*n2) != 0)):
                                newAcurracyValueToAdd = (1-(abs(n2-n1)/abs(n2)))
                                if (newAcurracyValueToAdd < 0):
                                    newAcurracyValueToAdd = 0
                                predictionAcurracy = predictionAcurracy + newAcurracyValueToAdd
                        if (isClassification == True):
                            if (abs(n1) > abs(n2)): # n2 has to be the one with the highest value with respect to n1
                                n2 = predictedData[row][0]
                                n1 = matrix_y[row][0]
                            if ((n1==0) and (n2>=-1 and n2<=1) and (n2!=0)):
                                predictionAcurracy = predictionAcurracy + ((1-abs(n2))/(1-n1))
                            if (n1==n2):
                                predictionAcurracy = predictionAcurracy + 1
                    predictionAcurracy = predictionAcurracy/numberOfDataPoints*100
                    temporalRow = []
                    temporalRow.append(predictionAcurracy)
                    temporalRow.append(currentMatrix_b)
                    temporalRow.append(currentMatrix_x)
                    allAccuracies.append(temporalRow)
        
                    # We save the current the modeling results if they are better than
                    # the actual best
                    currentBestAccuracy = bestModelingResults[1]
                    if (predictionAcurracy > currentBestAccuracy):
                        bestModelingResults = []
                        bestModelingResults.append(currentMatrix_b)
                        bestModelingResults.append(predictionAcurracy)
                        bestModelingResults.append(predictedData)
                        bestModelingResults.append("Coefficients distribution is as follows: y = bo + b1*x1 + b2*x1^2 + b3*x2 + b4*x2^2 + b5*x1*x2")
            
        # We include all the reports of all the models studied to the reporting
        # variable that contains the report of the best fitting model and we
        # then return it
        bestModelingResults.append(allAccuracies)
        return bestModelingResults
    
    """
    getPolynomialRegression(
            orderOfThePolynomial = "whole number to represent the desired order of the polynomial model to find",
            evtfbmip="True to indicate to Eliminate Variables To Find Better Model If Possible. False if the contrary is desired",
            isClassification="set to True if you are solving a classification problem. False if otherwise")
    
    Returns the best fitting model of a regression problem that has only 1
    independent variable (x) in it, through a polynomial regression solution.
    
    EXAMPLE CODE:
        matrix_y = [
                [3.4769e-11],
                [7.19967e-11],
                [1.59797e-10],
                [3.79298e-10]
                ]
        matrix_x = [
                [-0.7],
                [-0.65],
                [-0.6],
                [-0.55]
                ]
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        regression = mSL.Regression(matrix_x, matrix_y)
        # "orderOfThePolynomial" = "whole number to represent the desired order of the polynomial model to find"
        # "evtfbmip" stands for "Eliminate Variables To Find Better Model If Possible"
        modelingResults = regression.getPolynomialRegression(orderOfThePolynomial=3, evtfbmip=True, isClassification=False)
        modelCoefficients = modelingResults[0]
        acurracy = modelingResults[1]
        predictedData = modelingResults[2]
        coefficientDistribution = modelingResults[3]
        allModeledAccuracies = modelingResults[4]
        
    RESULT OF CODE:
        modelCoefficients =
        [[3.468869185343018e-08],
         [1.5123521825664843e-07],
         [2.2104758041867345e-07],
         [1.0817080022072073e-07]]
        
        acurracy =
        99.99999615014885
        
        predictedData =
        [[3.4769003219065136e-11],
         [7.199670288280337e-11],
         [1.597970024878988e-10],
         [3.792980021998557e-10]]
        
        coefficientDistribution =
        'Coefficients distribution is as follows: y = bo + b1*x + b2*x^2 + b3*x^3 + ... + bn*x^n'
        
        allModeledAccuracies["independent variable distribution used to get a model"]["model accuracy", "model coefficients obtained but with original distribution", "matrix x data"] =
        # NOTE: since this variable contains large amounts of information, it
        #       will not be displayed but only described on how to use it.
    """
    def getPolynomialRegression(self, orderOfThePolynomial, evtfbmip=False, isClassification=True):
        from ..linearAlgebra import MortrackLinearAlgebraLibrary as mLAL
        matrixMath = mLAL.MatrixMath()
        x_samples = matrixMath.getTransposedMatrix(self.x_samplesList)[0]
        y_samples = matrixMath.getTransposedMatrix(self.y_samplesList)[0]
        dataLength = len(y_samples)
        matrixLength = orderOfThePolynomial+1
        matrix_A = []
        # MATRIX A MATHEMATICAL PROCEDURE
        for n in range(0, matrixLength):
            temporalRow = []
            for i in range(0, matrixLength):
                # Math process for Matrix_A's Row 1
                if ((n==0) and (i==0)):
                    temporalRow.append(dataLength)
                if ((n==0) and (i!=0)):
                    temporalSum = 0
                    for j in range(0, dataLength):
                        # For loop use to get the x_i value elevated to an exponential
                        xMultiplicationsResult = 1
                        for w in range(0, i):
                            xMultiplicationsResult = xMultiplicationsResult*x_samples[j]
                        temporalSum = temporalSum + xMultiplicationsResult
                    temporalRow.append(temporalSum)
                # Math process for Matrix_A's Row 2 and above
                if (n!=0):
                    if (i==0):
                        temporalSum = 0
                        for j in range(0, dataLength):
                            # For loop use to get the x_i value elevated to an exponential
                            additionalMultiplications = n-1
                            if (additionalMultiplications < 0):
                                additionalMultiplications = 0
                            xMultiplicationsResult = 1
                            for w in range(0, (i+1+additionalMultiplications)):
                                xMultiplicationsResult = xMultiplicationsResult*x_samples[j]
                            temporalSum = temporalSum + xMultiplicationsResult
                        temporalRow.append(temporalSum)
                    else:
                        temporalSum = 0
                        for j in range(0, dataLength):
                            # For loop use to get the x_i value elevated to an exponential
                            additionalMultiplications = n-1
                            if (additionalMultiplications < 0):
                                additionalMultiplications = 0
                            xMultiplicationsResult = 1
                            for w in range(0, (i+1+additionalMultiplications)):
                                xMultiplicationsResult = xMultiplicationsResult*x_samples[j]
                            temporalSum = temporalSum + xMultiplicationsResult
                        temporalRow.append(temporalSum)     
            matrix_A.append(temporalRow)
        # MATRIX g MATHEMATICAL PROCEDURE
        matrix_g = []
        for n in range(0, matrixLength):
            temporalRow = []
            temporalSum = 0
            for i in range(0, dataLength):
                # For loop use to get the x_i value elevated to an exponential
                xMultiplicationsResult = 1
                for w in range(0, n):
                    xMultiplicationsResult = xMultiplicationsResult*x_samples[i]
                temporalSum = temporalSum + xMultiplicationsResult*y_samples[i]
            temporalRow.append(temporalSum)
            matrix_g.append(temporalRow)
        # GET THE INVERSE OF MATRIX A
        matrixMath = mLAL.MatrixMath()
        inverseMatrix_A = matrixMath.getInverse(matrix_A)
        # MULTIPLY INVERSE OF MATRIX A WITH MATRIX g
        matrix_b = matrixMath.getMultiplication(inverseMatrix_A, matrix_g)

        # We determine the accuracy of the obtained coefficients
        predictedData = []
        bothMatrixRowLength = len(y_samples)
        numberOfIndependentVariables = len(matrix_b)-1
        for currentDataPoint in range(0, bothMatrixRowLength):
            temporalRow = []
            actualIc = matrix_b[0][0]
            for currentIndependentVariable in range(0, numberOfIndependentVariables):
                actualIc = actualIc + matrix_b[currentIndependentVariable+1][0]*x_samples[currentDataPoint]**(currentIndependentVariable+1)
            temporalRow.append(actualIc)
            predictedData.append(temporalRow)
        predictionAcurracy = 0
        numberOfDataPoints = bothMatrixRowLength
        for row in range(0, numberOfDataPoints):
            n2 = self.y_samplesList[row][0]
            n1 = predictedData[row][0]
            if (isClassification == False):
                if (((n1*n2) != 0)):
                    newAcurracyValueToAdd = (1-(abs(n2-n1)/abs(n2)))
                    if (newAcurracyValueToAdd < 0):
                        newAcurracyValueToAdd = 0
                    predictionAcurracy = predictionAcurracy + newAcurracyValueToAdd
            if (isClassification == True):
                if (abs(n1) > abs(n2)): # n2 has to be the one with the highest value with respect to n1
                    n2 = predictedData[row][0]
                    n1 = self.y_samplesList[row][0]
                if ((n1==0) and (n2>=-1 and n2<=1) and (n2!=0)):
                    predictionAcurracy = predictionAcurracy + ((1-abs(n2))/(1-n1))
                if (n1==n2):
                    predictionAcurracy = predictionAcurracy + 1
        predictionAcurracy = predictionAcurracy/numberOfDataPoints*100
        
        # We save the current the modeling results
        bestModelingResults = []
        bestModelingResults.append(matrix_b)
        bestModelingResults.append(predictionAcurracy)
        bestModelingResults.append(predictedData)
        bestModelingResults.append("Coefficients distribution is as follows: y = bo + b1*x + b2*x^2 + b3*x^3 + ... + bn*x^n")
        allAccuracies = []
        temporalRow = []
        temporalRow.append(bestModelingResults[1])
        temporalRow.append(bestModelingResults[0])
        temporalRow.append(self.x_samplesList)
        allAccuracies.append(temporalRow)
        bestModelingResults.append(allAccuracies)
        
        # We recreate some things to apply the Matrix method in the permutation
        # section
        rowLengthOfBothMatrixes = len(self.y_samplesList)
        currentMatrix_x = []
        for row in range(0, rowLengthOfBothMatrixes):
            temporalRow = []
            temporalRow.append(1)
            for currentIndependentVariable in range(0, numberOfIndependentVariables):
                temporalRow.append(self.x_samplesList[row][0]**(currentIndependentVariable+1))
            currentMatrix_x.append(temporalRow)
        originalMatrix_x = currentMatrix_x
        from .MortrackML_Library import Combinations
        possibleCombinations = []
        for n in range (0, len(originalMatrix_x[0])):
            possibleCombinations.append(n)
        combinations = Combinations(possibleCombinations)
        
        if (evtfbmip == True):
            # ----------------------------------------------------------------------------------------------- #
            # ----- We now get all possible combinations/permutations with the elements of our equation ----- #
            # ----------------------------------------------------------------------------------------------- #
            customizedPermutations = combinations.getCustomizedPermutationList()
            customizedPermutations.pop(0) # We remove the null value
            customizedPermutations.pop(len(customizedPermutations)-1) # We remove the last one because we already did it
            for actualPermutation in range(0, len(customizedPermutations)):
                newOriginalMatrix_x = []
                for row in range(0, rowLengthOfBothMatrixes):
                    temporalRow = []
                    for column in range(0, len(customizedPermutations[actualPermutation])):
                        temporalRow.append(originalMatrix_x[row][customizedPermutations[actualPermutation][column]])
                    newOriginalMatrix_x.append(temporalRow)
                
            # ----- WE START SEARCHING FOR THE BEST MODELING RESULTS USING CURRENT PERMUTATION ----- #
                # We define a variable to save the search patterns in original matrix x
                possibleCombinations = []
                for n in range (0, len(newOriginalMatrix_x[0])):
                    possibleCombinations.append(n)
                combinations = Combinations(possibleCombinations)
                searchPatterns = combinations.getPositionCombinationsList()
                
                # We start to search for the coefficients that give us the best accuracy
                for currentSearchPattern in range(0, len(searchPatterns)):
                    currentMatrix_x = [ [ 0 for i in range(len(newOriginalMatrix_x[0])) ] for j in range(rowLengthOfBothMatrixes) ]
                    # We assign the current distribution that we want to study of the
                    # variables of the matrix x, to evaluate its resulting regression
                    # coefficients
                    for currentColumnOfMatrix_x in range(0, len(newOriginalMatrix_x[0])):
                        for column in range(0, len(newOriginalMatrix_x[0])):
                            if (customizedPermutations[actualPermutation][searchPatterns[currentSearchPattern][currentColumnOfMatrix_x]] == column):
                                for row in range(0, rowLengthOfBothMatrixes):
                                    currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][column]
                    # We get the Transposed matrix of matrix X. NOTE: We create a temporal
                    # variable to save matrix x because remember that in python, children
                    # and parent inheritance is ignored when using clases
                    temporalMatrix1 = currentMatrix_x
                    transposedMatrix_X = matrixMath.getTransposedMatrix(temporalMatrix1)
                    # WE GET MATRIX A.  NOTE: We create a temporal
                    # variable to save matrix x because remember that in python, children
                    # and parent inheritance is ignored when using clases
                    temporalMatrix1 = transposedMatrix_X
                    temporalMatrix2 = currentMatrix_x
                    matrix_A = matrixMath.getMultiplication(temporalMatrix1, temporalMatrix2)
                    # WE GET MATRIX g. NOTE: We create a temporal
                    # variable to save matrix x because remember that in python, children
                    # and parent inheritance is ignored when using clases
                    temporalMatrix1 = transposedMatrix_X
                    temporalMatrix2 = self.y_samplesList
                    matrix_g = matrixMath.getMultiplication(temporalMatrix1, temporalMatrix2)
                    # We get inverse matrix of matrix A.
                    inversedMatrix_A = matrixMath.getInverse(matrix_A)
                    # We get matrix b, which will contain the coeficient values
                    matrix_b = matrixMath.getMultiplication(inversedMatrix_A, matrix_g)
                    
                    # ----- WE DETERMINE THE ACCURACY OF THE OBTAINED COEFFICIENTS ----- #
                    # We re-arrange the obtained coefficients to then evaluate this
                    # model
                    currentMatrix_b = [ [ 0 for i in range(1) ] for j in range(len(originalMatrix_x[0])) ]
                    for row in range(0, len(newOriginalMatrix_x[0])):
                        trueRowOfCoefficient = customizedPermutations[actualPermutation][searchPatterns[currentSearchPattern][row]]
                        currentMatrix_b[trueRowOfCoefficient][0] = matrix_b[row][0]
                    # We obtain the predicted data through the current obtained
                    # coefficients
                    newNumberOfIndependentVariables = len(currentMatrix_x[0])
                    predictedData = []
                    for row in range(0, len(self.y_samplesList)):
                        temporalRow = []
                        actualIc = currentMatrix_b[0][0]
                        for currentIndependentVariable in range(0, (newNumberOfIndependentVariables-1)):
                            actualIc = actualIc + currentMatrix_b[currentIndependentVariable+1][0]*self.x_samplesList[row][0]**(currentIndependentVariable+1)
                        temporalRow.append(actualIc)
                        predictedData.append(temporalRow)
                    predictionAcurracy = 0
                    numberOfDataPoints = len(self.y_samplesList)
                    for row in range(0, numberOfDataPoints):
                        n2 = self.y_samplesList[row][0]
                        n1 = predictedData[row][0]
                        if (isClassification == False):
                            if (((n1*n2) != 0)):
                                newAcurracyValueToAdd = (1-(abs(n2-n1)/abs(n2)))
                                if (newAcurracyValueToAdd < 0):
                                    newAcurracyValueToAdd = 0
                                predictionAcurracy = predictionAcurracy + newAcurracyValueToAdd
                        if (isClassification == True):
                            if (abs(n1) > abs(n2)): # n2 has to be the one with the highest value with respect to n1
                                n2 = predictedData[row][0]
                                n1 = self.y_samplesList[row][0]
                            if ((n1==0) and (n2>=-1 and n2<=1) and (n2!=0)):
                                predictionAcurracy = predictionAcurracy + ((1-abs(n2))/(1-n1))
                            if (n1==n2):
                                predictionAcurracy = predictionAcurracy + 1
                    predictionAcurracy = predictionAcurracy/numberOfDataPoints*100
                    temporalRow = []
                    temporalRow.append(predictionAcurracy)
                    temporalRow.append(currentMatrix_b)
                    temporalRow.append(currentMatrix_x)
                    allAccuracies.append(temporalRow)
                    
                    # We save the current the modeling results if they are better than
                    # the actual best
                    currentBestAccuracy = bestModelingResults[1]           
                    if (predictionAcurracy > currentBestAccuracy):
                        bestModelingResults = []
                        bestModelingResults.append(currentMatrix_b)
                        bestModelingResults.append(predictionAcurracy)
                        bestModelingResults.append(predictedData)
                        bestModelingResults.append("Coefficients distribution is as follows: y = bo + b1*x + b2*x^2 + b3*x^3 + ... + bn*x^n")
            
        # We include all the reports of all the models studied to the reporting
        # variable that contains the report of the best fitting model and we
        # then return it
        bestModelingResults.append(allAccuracies)
        return bestModelingResults
    
    """
    getMultiplePolynomialRegression(
            orderOfThePolynomial = "whole number to represent the desired order of the polynomial model to find",
            evtfbmip="True to indicate to Eliminate Variables To Find Better Model If Possible. False if the contrary is desired",
            isClassification="set to True if you are solving a classification problem. False if otherwise")
    
    This method returns the best fitting model of a dataset to predict its
    behavior through a Multiple Polynomial Regression that may have any number
    of independent variables (x). This method gets a model by through the
    following equation format:
        y = bo + b1*x1 + b2*x1^2 + ... + bn*x1^n + b3*x2 + b4*x2^2 + ... + bn*x2^n + b5*x3 + b6*x3^2 + ... + bn*xn^n
    
    CODE EXAMPLE:
        matrix_y = [
                [14.05],
                [10.55],
                [7.55],
                [14.93],
                [9.48],
                [6.59],
                [16.56],
                [13.63],
                [9.23],
                [15.85],
                [11.75],
                [8.78],
                [22.41],
                [18.55],
                [15.93],
                [21.66],
                [17.98],
                [16.44]
                ]
        matrix_x = [
                [75, 15],
                [100, 15],
                [125, 15],
                [75, 17.5],
                [100, 17.5],
                [125, 17.5],
                [75, 20],
                [100, 20],
                [125, 20],
                [75, 22.5],
                [100, 22.5],
                [125, 22.5],
                [75, 25],
                [100, 25],
                [125, 25],
                [75, 27.5],
                [100, 27.5],
                [125, 27.5]
                ]
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        regression = mSL.Regression(matrix_x, matrix_y)
        # "orderOfThePolynomial" = "whole number to represent the desired order of the polynomial model to find"
        # "evtfbmip" stands for "Eliminate Variables To Find Better Model If Possible"
        modelingResults = regression.getMultiplePolynomialRegression(orderOfThePolynomial=4, evtfbmip=True, isClassification=False)
        modelCoefficients = modelingResults[0]
        acurracy = modelingResults[1]
        predictedData = modelingResults[2]
        coefficientDistribution = modelingResults[3]
        allModeledAccuracies = modelingResults[4]
        
    RESULT OF CODE:
        modelCoefficients =
        [[-1.745717777706403e-08],
         [0],
         [0.07581354676648289],
         [-0.00104662847289827],
         [3.942075523087618e-06],
         [-14.202436859894078],
         [0.670002091817878],
         [-0.009761974914994198],
         [-5.8006065221068606e-15]]
        
        acurracy =
        91.33822971744071
        
        predictedData =
        [[14.401799310251064],
         [10.481799480368835],
         [7.578466505722503],
         [13.96195814877683],
         [10.041958318894615],
         [7.1386253442482825],
         [15.490847097061135],
         [11.57084726717892],
         [8.667514292532587],
         [18.073281006823265],
         [14.15328117694105],
         [11.249948202294718],
         [20.794074729782523],
         [16.874074899900307],
         [13.970741925253975],
         [22.73804311765818],
         [18.818043287775964],
         [15.914710313129632]]
        
        coefficientDistribution =
        'Coefficients distribution is as follows: y = bo + b1*x1 + b2*x1^2 + ... + bn*x1^n + b3*x2 + b4*x2^2 + ... + bn*x2^n + b5*x3 + b6*x3^2 + ... + bn*xn^n'
        
        allModeledAccuracies["independent variable distribution used to get a model"]["model accuracy", "model coefficients obtained but with original distribution", "matrix x data"] =
        # NOTE: since this variable contains large amounts of information, it
        #       will not be displayed but only described on how to use it.
    """
    def getMultiplePolynomialRegression(self, orderOfThePolynomial, evtfbmip=False, isClassification=True):
        # We import the libraries we want to use and we create the class we
        # use from it
        from ..linearAlgebra import MortrackLinearAlgebraLibrary as mLAL
        matrixMath = mLAL.MatrixMath()
        # We define the variables to use within our code
        numberOfIndependentVariables = len(self.x_samplesList[0])
        rowLengthOfBothMatrixes = len(self.y_samplesList)
        matrix_x = []
        # MATRIX X MATHEMATICAL PROCEDURE to add the 1's in the first column of
        # each row and to add the additional columns that will represent the
        # polynomials that we want to get according to the input value of
        # this method's argument "orderOfThePolynomial"
        for row in range(0, rowLengthOfBothMatrixes):
            temporalRow = []
            temporalRow.append(1)
            for actualIndependentVariable in range(0, numberOfIndependentVariables):
                xMultiplicationsResult = 1
                for actualOrder in range(0, orderOfThePolynomial):
                    xMultiplicationsResult = xMultiplicationsResult*self.x_samplesList[row][actualIndependentVariable]
                    temporalRow.append(xMultiplicationsResult)
            matrix_x.append(temporalRow)
        originalMatrix_x = matrix_x
        # We get the Transposed matrix of matrix X. NOTE: We create a temporal
        # variable to save matrix x because remember that in python, children
        # and parent inheritance is ignored when using clases
        temporalMatrix1 = matrix_x
        transposedMatrix_X = matrixMath.getTransposedMatrix(temporalMatrix1)
        # WE GET MATRIX A.  NOTE: We create a temporal
        # variable to save matrix x because remember that in python, children
        # and parent inheritance is ignored when using clases
        temporalMatrix1 = transposedMatrix_X
        temporalMatrix2 = matrix_x
        matrix_A = matrixMath.getMultiplication(temporalMatrix1, temporalMatrix2)
        # WE GET MATRIX g. NOTE: We create a temporal
        # variable to save matrix x because remember that in python, children
        # and parent inheritance is ignored when using clases
        temporalMatrix1 = transposedMatrix_X
        temporalMatrix2 = self.y_samplesList
        matrix_g = matrixMath.getMultiplication(temporalMatrix1, temporalMatrix2)
        # We get inverse matrix of matrix A.
        inversedMatrix_A = matrixMath.getInverse(matrix_A)
        # We get matrix b, which will contain the coeficient values
        matrix_b = matrixMath.getMultiplication(inversedMatrix_A, matrix_g)
        
        # We determine the accuracy of the obtained coefficients
        predictedData = []
        numberOfCoefficients = len(matrix_b)
        for row in range(0, rowLengthOfBothMatrixes):
            temporalRow = []
            actualIc = matrix_b[0][0]
            currentOrderOfThePolynomial = 1
            currentVariable = 0
            for currentIndependentVariable in range(0, numberOfCoefficients-1):
                if (currentOrderOfThePolynomial == (orderOfThePolynomial+1)):
                    currentOrderOfThePolynomial = 1
                    currentVariable = currentVariable + 1
                actualIc = actualIc + matrix_b[currentIndependentVariable+1][0]*self.x_samplesList[row][currentVariable]**(currentOrderOfThePolynomial)
                currentOrderOfThePolynomial = currentOrderOfThePolynomial + 1
            temporalRow.append(actualIc)
            predictedData.append(temporalRow)
        predictionAcurracy = 0
        numberOfDataPoints = len(self.y_samplesList)
        for row in range(0, numberOfDataPoints):
            n2 = self.y_samplesList[row][0]
            n1 = predictedData[row][0]
            if (isClassification == False):
                if (((n1*n2) != 0)):
                    newAcurracyValueToAdd = (1-(abs(n2-n1)/abs(n2)))
                    if (newAcurracyValueToAdd < 0):
                        newAcurracyValueToAdd = 0
                    predictionAcurracy = predictionAcurracy + newAcurracyValueToAdd
            if (isClassification == True):
                if (abs(n1) > abs(n2)): # n2 has to be the one with the highest value with respect to n1
                    n2 = predictedData[row][0]
                    n1 = self.y_samplesList[row][0]
                if ((n1==0) and (n2>=-1 and n2<=1) and (n2!=0)):
                    predictionAcurracy = predictionAcurracy + ((1-abs(n2))/(1-n1))
                if (n1==n2):
                    predictionAcurracy = predictionAcurracy + 1
        predictionAcurracy = predictionAcurracy/numberOfDataPoints*100
        
        # We save the current the modeling results
        bestModelingResults = []
        bestModelingResults.append(matrix_b)
        bestModelingResults.append(predictionAcurracy)
        bestModelingResults.append(predictedData)
        bestModelingResults.append("Coefficients distribution is as follows: y = bo + b1*x1 + b2*x1^2 + ... + bn*x1^n + b3*x2 + b4*x2^2 + ... + bn*x2^n + b5*x3 + b6*x3^2 + ... + bn*xn^n")
        allAccuracies = []
        temporalRow = []
        temporalRow.append(bestModelingResults[1])
        temporalRow.append(bestModelingResults[0])
        temporalRow.append(originalMatrix_x)
        allAccuracies.append(temporalRow)
        
        # ----- WE START SEARCHING FOR THE BEST MODELING RESULTS ----- #
        # We define a variable to save the search patterns in original matrix x
        from .MortrackML_Library import Combinations
        possibleCombinations = []
        for n in range (0, len(originalMatrix_x[0])):
            possibleCombinations.append(n)
        combinations = Combinations(possibleCombinations)
        searchPatterns = combinations.getPositionCombinationsList()
        searchPatterns.pop(0) # We remove the first one because we already did it
        # We start to search for the coefficients that give us the best accuracy
        for currentSearchPattern in range(0, len(searchPatterns)):
            currentMatrix_x = [ [ 0 for i in range(len(originalMatrix_x[0])) ] for j in range(rowLengthOfBothMatrixes) ]
            # We assign the current distribution that we want to study of the
            # variables of the matrix x, to evaluate its resulting regression
            # coefficients
            for currentColumnOfMatrix_x in range(0, len(originalMatrix_x[0])):
                for column in range(0, len(originalMatrix_x[0])):
                    if (searchPatterns[currentSearchPattern][currentColumnOfMatrix_x] == column):
                        for row in range(0, rowLengthOfBothMatrixes):
                            currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][column]
            # We get the Transposed matrix of matrix X. NOTE: We create a temporal
            # variable to save matrix x because remember that in python, children
            # and parent inheritance is ignored when using clases
            temporalMatrix1 = currentMatrix_x
            transposedMatrix_X = matrixMath.getTransposedMatrix(temporalMatrix1)
            # WE GET MATRIX A.  NOTE: We create a temporal
            # variable to save matrix x because remember that in python, children
            # and parent inheritance is ignored when using clases
            temporalMatrix1 = transposedMatrix_X
            temporalMatrix2 = currentMatrix_x
            matrix_A = matrixMath.getMultiplication(temporalMatrix1, temporalMatrix2)
            # WE GET MATRIX g. NOTE: We create a temporal
            # variable to save matrix x because remember that in python, children
            # and parent inheritance is ignored when using clases
            temporalMatrix1 = transposedMatrix_X
            temporalMatrix2 = self.y_samplesList
            matrix_g = matrixMath.getMultiplication(temporalMatrix1, temporalMatrix2)
            # We get inverse matrix of matrix A.
            inversedMatrix_A = matrixMath.getInverse(matrix_A)
            # We get matrix b, which will contain the coeficient values
            matrix_b = matrixMath.getMultiplication(inversedMatrix_A, matrix_g)
            
            # ----- WE DETERMINE THE ACCURACY OF THE OBTAINED COEFFICIENTS ----- #
            # We re-arrange the obtained coefficients to then evaluate this
            # model
            currentMatrix_b = [ [ 0 for i in range(1) ] for j in range(len(originalMatrix_x[0])) ]
            for row in range(0, len(originalMatrix_x[0])):
                trueRowOfCoefficient = searchPatterns[currentSearchPattern][row]
                currentMatrix_b[trueRowOfCoefficient][0] = matrix_b[row][0]
            # We obtain the predicted data through the current obtained
            # coefficients
            numberOfCoefficients = len(currentMatrix_b)
            predictedData = []
            for row in range(0, len(self.y_samplesList)):
                temporalRow = []
                actualIc = currentMatrix_b[0][0]
                currentOrderOfThePolynomial = 1
                currentVariable = 0
                for currentIndependentVariable in range(0, numberOfCoefficients-1):
                    if (currentOrderOfThePolynomial == (orderOfThePolynomial+1)):
                        currentOrderOfThePolynomial = 1
                        currentVariable = currentVariable + 1
                    actualIc = actualIc + currentMatrix_b[currentIndependentVariable+1][0]*self.x_samplesList[row][currentVariable]**(currentOrderOfThePolynomial)
                    currentOrderOfThePolynomial = currentOrderOfThePolynomial + 1
                temporalRow.append(actualIc)
                predictedData.append(temporalRow)
            predictionAcurracy = 0
            numberOfDataPoints = len(self.y_samplesList)
            for row in range(0, numberOfDataPoints):
                n2 = self.y_samplesList[row][0]
                n1 = predictedData[row][0]
                if (isClassification == False):
                    if (((n1*n2) != 0)):
                        newAcurracyValueToAdd = (1-(abs(n2-n1)/abs(n2)))
                        if (newAcurracyValueToAdd < 0):
                            newAcurracyValueToAdd = 0
                        predictionAcurracy = predictionAcurracy + newAcurracyValueToAdd
                if (isClassification == True):
                    if (abs(n1) > abs(n2)): # n2 has to be the one with the highest value with respect to n1
                        n2 = predictedData[row][0]
                        n1 = self.y_samplesList[row][0]
                    if ((n1==0) and (n2>=-1 and n2<=1) and (n2!=0)):
                        predictionAcurracy = predictionAcurracy + ((1-abs(n2))/(1-n1))
                    if (n1==n2):
                        predictionAcurracy = predictionAcurracy + 1
            predictionAcurracy = predictionAcurracy/numberOfDataPoints*100
            temporalRow = []
            temporalRow.append(predictionAcurracy)
            temporalRow.append(currentMatrix_b)
            temporalRow.append(currentMatrix_x)
            allAccuracies.append(temporalRow)

            # We save the current the modeling results if they are better than
            # the actual best
            currentBestAccuracy = bestModelingResults[1]
            if (predictionAcurracy > currentBestAccuracy):
                bestModelingResults = []
                bestModelingResults.append(currentMatrix_b)
                bestModelingResults.append(predictionAcurracy)
                bestModelingResults.append(predictedData)
                bestModelingResults.append("Coefficients distribution is as follows: y = bo + b1*x1 + b2*x1^2 + ... + bn*x1^n + b3*x2 + b4*x2^2 + ... + bn*x2^n + b5*x3 + b6*x3^2 + ... + bn*xn^n")
        if (evtfbmip == True):
            # ----------------------------------------------------------------------------------------------- #
            # ----- We now get all possible combinations/permutations with the elements of our equation ----- #
            # ----------------------------------------------------------------------------------------------- #
            customizedPermutations = combinations.getCustomizedPermutationList()
            customizedPermutations.pop(0) # We remove the null value
            customizedPermutations.pop(len(customizedPermutations)-1) # We remove the last one because we already did it
            for actualPermutation in range(0, len(customizedPermutations)):
                newOriginalMatrix_x = []
                for row in range(0, rowLengthOfBothMatrixes):
                    temporalRow = []
                    for column in range(0, len(customizedPermutations[actualPermutation])):
                        temporalRow.append(originalMatrix_x[row][customizedPermutations[actualPermutation][column]])
                    newOriginalMatrix_x.append(temporalRow)
                
            # ----- WE START SEARCHING FOR THE BEST MODELING RESULTS USING CURRENT PERMUTATION ----- #
                # We define a variable to save the search patterns in original matrix x
                possibleCombinations = []
                for n in range (0, len(newOriginalMatrix_x[0])):
                    possibleCombinations.append(n)
                combinations = Combinations(possibleCombinations)
                searchPatterns = combinations.getPositionCombinationsList()
                
                # We start to search for the coefficients that give us the best accuracy
                for currentSearchPattern in range(0, len(searchPatterns)):
                    currentMatrix_x = [ [ 0 for i in range(len(newOriginalMatrix_x[0])) ] for j in range(rowLengthOfBothMatrixes) ]
                    # We assign the current distribution that we want to study of the
                    # variables of the matrix x, to evaluate its resulting regression
                    # coefficients
                    for currentColumnOfMatrix_x in range(0, len(newOriginalMatrix_x[0])):
                        for column in range(0, len(newOriginalMatrix_x[0])):
                            if (customizedPermutations[actualPermutation][searchPatterns[currentSearchPattern][currentColumnOfMatrix_x]] == column):
                                for row in range(0, rowLengthOfBothMatrixes):
                                    currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][column]
                    # We get the Transposed matrix of matrix X. NOTE: We create a temporal
                    # variable to save matrix x because remember that in python, children
                    # and parent inheritance is ignored when using clases
                    temporalMatrix1 = currentMatrix_x
                    transposedMatrix_X = matrixMath.getTransposedMatrix(temporalMatrix1)
                    # WE GET MATRIX A.  NOTE: We create a temporal
                    # variable to save matrix x because remember that in python, children
                    # and parent inheritance is ignored when using clases
                    temporalMatrix1 = transposedMatrix_X
                    temporalMatrix2 = currentMatrix_x
                    matrix_A = matrixMath.getMultiplication(temporalMatrix1, temporalMatrix2)
                    # WE GET MATRIX g. NOTE: We create a temporal
                    # variable to save matrix x because remember that in python, children
                    # and parent inheritance is ignored when using clases
                    temporalMatrix1 = transposedMatrix_X
                    temporalMatrix2 = self.y_samplesList
                    matrix_g = matrixMath.getMultiplication(temporalMatrix1, temporalMatrix2)
                    # We get inverse matrix of matrix A.
                    inversedMatrix_A = matrixMath.getInverse(matrix_A)
                    # We get matrix b, which will contain the coeficient values
                    matrix_b = matrixMath.getMultiplication(inversedMatrix_A, matrix_g)
                    
                    # ----- WE DETERMINE THE ACCURACY OF THE OBTAINED COEFFICIENTS ----- #
                    # We re-arrange the obtained coefficients to then evaluate this
                    # model
                    currentMatrix_b = [ [ 0 for i in range(1) ] for j in range(len(originalMatrix_x[0])) ]
                    for row in range(0, len(newOriginalMatrix_x[0])):
                        trueRowOfCoefficient = customizedPermutations[actualPermutation][searchPatterns[currentSearchPattern][row]]
                        currentMatrix_b[trueRowOfCoefficient][0] = matrix_b[row][0]
                    # We obtain the predicted data through the current obtained
                    # coefficients
                    predictedData = []
                    numberOfCoefficients = len(currentMatrix_b)
                    for row in range(0, len(self.y_samplesList)):
                        temporalRow = []
                        actualIc = currentMatrix_b[0][0]
                        currentOrderOfThePolynomial = 1
                        currentVariable = 0
                        for currentIndependentVariable in range(0, numberOfCoefficients-1):
                            if (currentOrderOfThePolynomial == (orderOfThePolynomial+1)):
                                currentOrderOfThePolynomial = 1
                                currentVariable = currentVariable + 1
                            actualIc = actualIc + currentMatrix_b[currentIndependentVariable+1][0]*self.x_samplesList[row][currentVariable]**(currentOrderOfThePolynomial)
                            currentOrderOfThePolynomial = currentOrderOfThePolynomial + 1
                        temporalRow.append(actualIc)
                        predictedData.append(temporalRow)
                    
                    predictionAcurracy = 0
                    numberOfDataPoints = len(self.y_samplesList)
                    for row in range(0, numberOfDataPoints):
                        n2 = self.y_samplesList[row][0]
                        n1 = predictedData[row][0]
                        if (isClassification == False):
                            if (((n1*n2) != 0)):
                                newAcurracyValueToAdd = (1-(abs(n2-n1)/abs(n2)))
                                if (newAcurracyValueToAdd < 0):
                                    newAcurracyValueToAdd = 0
                                predictionAcurracy = predictionAcurracy + newAcurracyValueToAdd
                        if (isClassification == True):
                            if (abs(n1) > abs(n2)): # n2 has to be the one with the highest value with respect to n1
                                n2 = predictedData[row][0]
                                n1 = self.y_samplesList[row][0]
                            if ((n1==0) and (n2>=-1 and n2<=1) and (n2!=0)):
                                predictionAcurracy = predictionAcurracy + ((1-abs(n2))/(1-n1))
                            if (n1==n2):
                                predictionAcurracy = predictionAcurracy + 1
                    predictionAcurracy = predictionAcurracy/numberOfDataPoints*100
                    temporalRow = []
                    temporalRow.append(predictionAcurracy)
                    temporalRow.append(currentMatrix_b)
                    temporalRow.append(currentMatrix_x)
                    allAccuracies.append(temporalRow)
        
                    # We save the current the modeling results if they are better than
                    # the actual best
                    currentBestAccuracy = bestModelingResults[1]
                    if (predictionAcurracy > currentBestAccuracy):
                        bestModelingResults = []
                        bestModelingResults.append(currentMatrix_b)
                        bestModelingResults.append(predictionAcurracy)
                        bestModelingResults.append(predictedData)
                        bestModelingResults.append("Coefficients distribution is as follows: y = bo + b1*x1 + b2*x1^2 + ... + bn*x1^n + b3*x2 + b4*x2^2 + ... + bn*x2^n + b5*x3 + b6*x3^2 + ... + bn*xn^n")
        # Alongside the information of the best model obtained, we add the
        # modeled information of ALL the models obtained to the variable that
        # we will return in this method
        bestModelingResults.append(allAccuracies)
        return bestModelingResults
        
    """
    getCustomizedMultipleSecondOrderPolynomialRegression(evtfbmip="True to indicate to Eliminate Variables To Find Better Model If Possible. False if the contrary is desired",
                                                         isClassification="set to True if you are solving a classification problem. False if otherwise")
    
    This method obtains the best solution of a customized 2nd order model when
    using specifically 2 independent variables and were the equation to solve
    is the following:
    y = bo + b1*x1 + b2*x1^2 + b3*x2 + b4*x2^2 + b5*x1*x2
    IMPORTANT NOTE: While the book "Probabilidad y estadistica para ingenieria
    & ciencias (Walpole, Myers, Myers, Ye)" describes a model whos accuracy is
    89.936% through finding a solution using the same model equation as used in
    this method, i was able to achieve a better algorithm that finds an even
    better solution were i was able to get an accuracy of 91.17% (see code
    example).
    
    CODE EXAMPLE:
        matrix_y = [
                [14.05],
                [10.55],
                [7.55],
                [14.93],
                [9.48],
                [6.59],
                [16.56],
                [13.63],
                [9.23],
                [15.85],
                [11.75],
                [8.78],
                [22.41],
                [18.55],
                [15.93],
                [21.66],
                [17.98],
                [16.44]
                ]
        matrix_x = [
                [75, 15],
                [100, 15],
                [125, 15],
                [75, 17.5],
                [100, 17.5],
                [125, 17.5],
                [75, 20],
                [100, 20],
                [125, 20],
                [75, 22.5],
                [100, 22.5],
                [125, 22.5],
                [75, 25],
                [100, 25],
                [125, 25],
                [75, 27.5],
                [100, 27.5],
                [125, 27.5]
                ]
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        regression = mSL.Regression(matrix_x, matrix_y)
        modelingResults = regression.getCustomizedMultipleSecondOrderPolynomialRegression(evtfbmip = True, isClassification=False)
        modelCoefficients = modelingResults[0]
        acurracy = modelingResults[1]
        predictedData = modelingResults[2]
        coefficientDistribution = modelingResults[3]
        allModeledAccuracies = modelingResults[4]
        
    RESULT OF CODE:
        modelCoefficients =
        [[40.36892063492269],
         [-0.29913333333337394],
         [0.0008133333333341963],
         [-1.2861238095233603],
         [0.047676190476181546],
         [0]]
        
        acurracy =
        90.56977726188016
        
        predictedData =
        [[13.944206349214937],
         [10.0242063492177],
         [7.120873015888202],
         [14.602587301596287],
         [10.68258730159905],
         [7.779253968269552],
         [15.856920634929907],
         [11.936920634932669],
         [9.033587301603172],
         [17.707206349215795],
         [13.787206349218557],
         [10.88387301588906],
         [20.153444444453953],
         [16.233444444456715],
         [13.330111111127216],
         [23.19563492064438],
         [19.275634920647143],
         [16.372301587317644]]
        
        coefficientDistribution =
        'Coefficients distribution is as follows: y = bo + b1*x1 + b2*x1^2 + b3*x2 + b4*x2^2 + b5*x1*x2'
        
        allModeledAccuracies["independent variable distribution used to get a model"]["model accuracy", "model coefficients obtained but with original distribution", "matrix x data"] =
        # NOTE: since this variable contains large amounts of information, it
        #       will not be displayed but only described on how to use it.
    """
    def getCustomizedMultipleSecondOrderPolynomialRegression(self, evtfbmip=False, isClassification=True):
        # We import the libraries we want to use and we create the class we
        # use from it
        from ..linearAlgebra import MortrackLinearAlgebraLibrary as mLAL
        matrixMath = mLAL.MatrixMath()
        # We define the variables to use within our code
        matrix_y =self.y_samplesList
        rowLengthOfBothMatrixes = len(matrix_y)
        x1 = 0
        x2 = 1
        # ----- WE GET THE FIRST MODEL EVALUATION RESULTS ----- #
        # MATRIX X MATHEMATICAL PROCEDURE to create a matrix that contains
        # the x values of the following equation we want to solve and in that
        # same variable formation:
        # y = bo + b1*x1 + b2*x1^2 + b3*x2 + b4*x2^2 + b5*x1*x2
        currentMatrix_x = []
        for row in range(0, rowLengthOfBothMatrixes):
            temporalRow = []
            temporalRow.append(1)
            temporalRow.append(self.x_samplesList[row][x1])
            temporalRow.append(self.x_samplesList[row][x1]**2)
            temporalRow.append(self.x_samplesList[row][x2])
            temporalRow.append(self.x_samplesList[row][x2]**2)
            temporalRow.append(self.x_samplesList[row][x1]*self.x_samplesList[row][x2])
            currentMatrix_x.append(temporalRow)
        originalMatrix_x = currentMatrix_x
            
        # We get the Transposed matrix of matrix X. NOTE: We create a temporal
        # variable to save matrix x because remember that in python, children
        # and parent inheritance is ignored when using clases
        temporalMatrix1 = currentMatrix_x
        transposedMatrix_X = matrixMath.getTransposedMatrix(temporalMatrix1)
        # WE GET MATRIX A.  NOTE: We create a temporal
        # variable to save matrix x because remember that in python, children
        # and parent inheritance is ignored when using clases
        temporalMatrix1 = transposedMatrix_X
        temporalMatrix2 = currentMatrix_x
        matrix_A = matrixMath.getMultiplication(temporalMatrix1, temporalMatrix2)
        # WE GET MATRIX g. NOTE: We create a temporal
        # variable to save matrix x because remember that in python, children
        # and parent inheritance is ignored when using clases
        temporalMatrix1 = transposedMatrix_X
        temporalMatrix2 = matrix_y
        matrix_g = matrixMath.getMultiplication(temporalMatrix1, temporalMatrix2)
        # We get inverse matrix of matrix A.
        inversedMatrix_A = matrixMath.getInverse(matrix_A)
        # We get matrix b, which will contain the coeficient values
        matrix_b = matrixMath.getMultiplication(inversedMatrix_A, matrix_g)
        
        # We determine the accuracy of the obtained coefficients
        predictedData = []
        for row in range(0, len(matrix_y)):
            temporalRow = []
            actualIc = matrix_b[0][0] + matrix_b[1][0]*self.x_samplesList[row][0] + matrix_b[2][0]*self.x_samplesList[row][0]**2 + matrix_b[3][0]*self.x_samplesList[row][1] + matrix_b[4][0]*self.x_samplesList[row][1]**2 + matrix_b[5][0]*self.x_samplesList[row][0]*self.x_samplesList[row][1]
            temporalRow.append(actualIc)
            predictedData.append(temporalRow)
        predictionAcurracy = 0
        numberOfDataPoints = len(matrix_y)
        for row in range(0, numberOfDataPoints):
            n2 = matrix_y[row][0]
            n1 = predictedData[row][0]
            if (isClassification == False):
                if (((n1*n2) != 0)):
                    newAcurracyValueToAdd = (1-(abs(n2-n1)/abs(n2)))
                    if (newAcurracyValueToAdd < 0):
                        newAcurracyValueToAdd = 0
                    predictionAcurracy = predictionAcurracy + newAcurracyValueToAdd
            if (isClassification == True):
                if (abs(n1) > abs(n2)): # n2 has to be the one with the highest value with respect to n1
                    n2 = predictedData[row][0]
                    n1 = matrix_y[row][0]
                if ((n1==0) and (n2>=-1 and n2<=1) and (n2!=0)):
                    predictionAcurracy = predictionAcurracy + ((1-abs(n2))/(1-n1))
                if (n1==n2):
                    predictionAcurracy = predictionAcurracy + 1
        predictionAcurracy = predictionAcurracy/numberOfDataPoints*100
        
        # We save the current the modeling results
        bestModelingResults = []
        bestModelingResults.append(matrix_b)
        bestModelingResults.append(predictionAcurracy)
        bestModelingResults.append(predictedData)
        bestModelingResults.append("Coefficients distribution is as follows: y = bo + b1*x1 + b2*x1^2 + b3*x2 + b4*x2^2 + b5*x1*x2")
        allAccuracies = []
        temporalRow = []
        temporalRow.append(bestModelingResults[1])
        temporalRow.append(bestModelingResults[0])
        temporalRow.append(originalMatrix_x)
        allAccuracies.append(temporalRow)
        
        # ----- WE START SEARCHING FOR THE BEST MODELING RESULTS ----- #
        # We define a variable to save the search patterns in original matrix x
        from .MortrackML_Library import Combinations
        possibleCombinations = []
        for n in range (0, len(originalMatrix_x[0])):
            possibleCombinations.append(n)
        combinations = Combinations(possibleCombinations)
        searchPatterns = combinations.getPositionCombinationsList()
        searchPatterns.pop(0) # We remove the first one because we already did it
        # We start to search for the coefficients that give us the best accuracy
        for currentSearchPattern in range(0, len(searchPatterns)):
            currentMatrix_x = [ [ 0 for i in range(len(originalMatrix_x[0])) ] for j in range(rowLengthOfBothMatrixes) ]
            # We assign the current distribution that we want to study of the
            # variables of the matrix x, to evaluate its resulting regression
            # coefficients
            for currentColumnOfMatrix_x in range(0, len(originalMatrix_x[0])):
                if (searchPatterns[currentSearchPattern][currentColumnOfMatrix_x] == 0):
                    for row in range(0, rowLengthOfBothMatrixes):
                        currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][0]
                if (searchPatterns[currentSearchPattern][currentColumnOfMatrix_x] == 1):
                    for row in range(0, rowLengthOfBothMatrixes):
                        currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][1]
                if (searchPatterns[currentSearchPattern][currentColumnOfMatrix_x] == 2):
                    for row in range(0, rowLengthOfBothMatrixes):
                        currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][2]
                if (searchPatterns[currentSearchPattern][currentColumnOfMatrix_x] == 3):
                    for row in range(0, rowLengthOfBothMatrixes):
                        currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][3]
                if (searchPatterns[currentSearchPattern][currentColumnOfMatrix_x] == 4):
                    for row in range(0, rowLengthOfBothMatrixes):
                        currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][4]
                if (searchPatterns[currentSearchPattern][currentColumnOfMatrix_x] == 5):
                    for row in range(0, rowLengthOfBothMatrixes):
                        currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][5]
            # We get the Transposed matrix of matrix X. NOTE: We create a temporal
            # variable to save matrix x because remember that in python, children
            # and parent inheritance is ignored when using clases
            temporalMatrix1 = currentMatrix_x
            transposedMatrix_X = matrixMath.getTransposedMatrix(temporalMatrix1)
            # WE GET MATRIX A.  NOTE: We create a temporal
            # variable to save matrix x because remember that in python, children
            # and parent inheritance is ignored when using clases
            temporalMatrix1 = transposedMatrix_X
            temporalMatrix2 = currentMatrix_x
            matrix_A = matrixMath.getMultiplication(temporalMatrix1, temporalMatrix2)
            # WE GET MATRIX g. NOTE: We create a temporal
            # variable to save matrix x because remember that in python, children
            # and parent inheritance is ignored when using clases
            temporalMatrix1 = transposedMatrix_X
            temporalMatrix2 = matrix_y
            matrix_g = matrixMath.getMultiplication(temporalMatrix1, temporalMatrix2)
            # We get inverse matrix of matrix A.
            inversedMatrix_A = matrixMath.getInverse(matrix_A)
            # We get matrix b, which will contain the coeficient values
            matrix_b = matrixMath.getMultiplication(inversedMatrix_A, matrix_g)
            
            # ----- WE DETERMINE THE ACCURACY OF THE OBTAINED COEFFICIENTS ----- #
            # We re-arrange the obtained coefficients to then evaluate this
            # model
            currentMatrix_b = [ [ 0 for i in range(1) ] for j in range(len(originalMatrix_x[0])) ]
            for row in range(0, len(originalMatrix_x[0])):
                trueRowOfCoefficient = searchPatterns[currentSearchPattern][row]
                currentMatrix_b[trueRowOfCoefficient][0] = matrix_b[row][0]
            # We obtain the predicted data through the current obtained
            # coefficients
            predictedData = []
            for row in range(0, len(matrix_y)):
                temporalRow = []
                actualIc = currentMatrix_b[0][0] + currentMatrix_b[1][0]*self.x_samplesList[row][0] + currentMatrix_b[2][0]*self.x_samplesList[row][0]**2 + currentMatrix_b[3][0]*self.x_samplesList[row][1] + currentMatrix_b[4][0]*self.x_samplesList[row][1]**2 + currentMatrix_b[5][0]*self.x_samplesList[row][0]*self.x_samplesList[row][1]
                temporalRow.append(actualIc)
                predictedData.append(temporalRow)
            
            predictionAcurracy = 0
            numberOfDataPoints = len(matrix_y)
            for row in range(0, numberOfDataPoints):
                n2 = matrix_y[row][0]
                n1 = predictedData[row][0]
                if (isClassification == False):
                    if (((n1*n2) != 0)):
                        newAcurracyValueToAdd = (1-(abs(n2-n1)/abs(n2)))
                        if (newAcurracyValueToAdd < 0):
                            newAcurracyValueToAdd = 0
                        predictionAcurracy = predictionAcurracy + newAcurracyValueToAdd
                if (isClassification == True):
                    if (abs(n1) > abs(n2)): # n2 has to be the one with the highest value with respect to n1
                        n2 = predictedData[row][0]
                        n1 = matrix_y[row][0]
                    if ((n1==0) and (n2>=-1 and n2<=1) and (n2!=0)):
                        predictionAcurracy = predictionAcurracy + ((1-abs(n2))/(1-n1))
                    if (n1==n2):
                        predictionAcurracy = predictionAcurracy + 1
            predictionAcurracy = predictionAcurracy/numberOfDataPoints*100
            temporalRow = []
            temporalRow.append(predictionAcurracy)
            temporalRow.append(currentMatrix_b)
            temporalRow.append(currentMatrix_x)
            allAccuracies.append(temporalRow)

            # We save the current the modeling results if they are better than
            # the actual best
            currentBestAccuracy = bestModelingResults[1]
            if (predictionAcurracy > currentBestAccuracy):
                bestModelingResults = []
                bestModelingResults.append(currentMatrix_b)
                bestModelingResults.append(predictionAcurracy)
                bestModelingResults.append(predictedData)
                bestModelingResults.append("Coefficients distribution is as follows: y = bo + b1*x1 + b2*x1^2 + b3*x2 + b4*x2^2 + b5*x1*x2")
        if (evtfbmip == True):
            # ----------------------------------------------------------------------------------------------- #
            # ----- We now get all possible combinations/permutations with the elements of our equation ----- #
            # ----------------------------------------------------------------------------------------------- #
            customizedPermutations = combinations.getCustomizedPermutationList()
            customizedPermutations.pop(0) # We remove the null value
            customizedPermutations.pop(len(customizedPermutations)-1) # We remove the last one because we already did it
            for actualPermutation in range(0, len(customizedPermutations)):
                newOriginalMatrix_x = []
                for row in range(0, rowLengthOfBothMatrixes):
                    temporalRow = []
                    for column in range(0, len(customizedPermutations[actualPermutation])):
                        temporalRow.append(originalMatrix_x[row][customizedPermutations[actualPermutation][column]])
                    newOriginalMatrix_x.append(temporalRow)
                
            # ----- WE START SEARCHING FOR THE BEST MODELING RESULTS USING CURRENT PERMUTATION ----- #
                # We define a variable to save the search patterns in original matrix x
                possibleCombinations = []
                for n in range (0, len(newOriginalMatrix_x[0])):
                    possibleCombinations.append(n)
                combinations = Combinations(possibleCombinations)
                searchPatterns = combinations.getPositionCombinationsList()
                
                # We start to search for the coefficients that give us the best accuracy
                for currentSearchPattern in range(0, len(searchPatterns)):
                    currentMatrix_x = [ [ 0 for i in range(len(newOriginalMatrix_x[0])) ] for j in range(rowLengthOfBothMatrixes) ]
                    # We assign the current distribution that we want to study of the
                    # variables of the matrix x, to evaluate its resulting regression
                    # coefficients
                    for currentColumnOfMatrix_x in range(0, len(newOriginalMatrix_x[0])):
                        if (customizedPermutations[actualPermutation][searchPatterns[currentSearchPattern][currentColumnOfMatrix_x]] == 0):
                            for row in range(0, rowLengthOfBothMatrixes):
                                currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][0]
                        if (customizedPermutations[actualPermutation][searchPatterns[currentSearchPattern][currentColumnOfMatrix_x]] == 1):
                            for row in range(0, rowLengthOfBothMatrixes):
                                currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][1]
                        if (customizedPermutations[actualPermutation][searchPatterns[currentSearchPattern][currentColumnOfMatrix_x]] == 2):
                            for row in range(0, rowLengthOfBothMatrixes):
                                currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][2]
                        if (customizedPermutations[actualPermutation][searchPatterns[currentSearchPattern][currentColumnOfMatrix_x]] == 3):
                            for row in range(0, rowLengthOfBothMatrixes):
                                currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][3]
                        if (customizedPermutations[actualPermutation][searchPatterns[currentSearchPattern][currentColumnOfMatrix_x]] == 4):
                            for row in range(0, rowLengthOfBothMatrixes):
                                currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][4]
                        if (customizedPermutations[actualPermutation][searchPatterns[currentSearchPattern][currentColumnOfMatrix_x]] == 5):
                            for row in range(0, rowLengthOfBothMatrixes):
                                currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][5]
                    # We get the Transposed matrix of matrix X. NOTE: We create a temporal
                    # variable to save matrix x because remember that in python, children
                    # and parent inheritance is ignored when using clases
                    temporalMatrix1 = currentMatrix_x
                    transposedMatrix_X = matrixMath.getTransposedMatrix(temporalMatrix1)
                    # WE GET MATRIX A.  NOTE: We create a temporal
                    # variable to save matrix x because remember that in python, children
                    # and parent inheritance is ignored when using clases
                    temporalMatrix1 = transposedMatrix_X
                    temporalMatrix2 = currentMatrix_x
                    matrix_A = matrixMath.getMultiplication(temporalMatrix1, temporalMatrix2)
                    # WE GET MATRIX g. NOTE: We create a temporal
                    # variable to save matrix x because remember that in python, children
                    # and parent inheritance is ignored when using clases
                    temporalMatrix1 = transposedMatrix_X
                    temporalMatrix2 = matrix_y
                    matrix_g = matrixMath.getMultiplication(temporalMatrix1, temporalMatrix2)
                    # We get inverse matrix of matrix A.
                    inversedMatrix_A = matrixMath.getInverse(matrix_A)
                    # We get matrix b, which will contain the coeficient values
                    matrix_b = matrixMath.getMultiplication(inversedMatrix_A, matrix_g)
                    
                    # ----- WE DETERMINE THE ACCURACY OF THE OBTAINED COEFFICIENTS ----- #
                    # We re-arrange the obtained coefficients to then evaluate this
                    # model
                    currentMatrix_b = [ [ 0 for i in range(1) ] for j in range(len(originalMatrix_x[0])) ]
                    for row in range(0, len(newOriginalMatrix_x[0])):
                        trueRowOfCoefficient = customizedPermutations[actualPermutation][searchPatterns[currentSearchPattern][row]]
                        currentMatrix_b[trueRowOfCoefficient][0] = matrix_b[row][0]
                    # We obtain the predicted data through the current obtained
                    # coefficients
                    predictedData = []
                    for row in range(0, len(matrix_y)):
                        temporalRow = []
                        actualIc = currentMatrix_b[0][0] + currentMatrix_b[1][0]*self.x_samplesList[row][0] + currentMatrix_b[2][0]*self.x_samplesList[row][0]**2 + currentMatrix_b[3][0]*self.x_samplesList[row][1] + currentMatrix_b[4][0]*self.x_samplesList[row][1]**2 + currentMatrix_b[5][0]*self.x_samplesList[row][0]*self.x_samplesList[row][1]
                        temporalRow.append(actualIc)
                        predictedData.append(temporalRow)
                    
                    predictionAcurracy = 0
                    numberOfDataPoints = len(matrix_y)
                    for row in range(0, numberOfDataPoints):
                        n2 = matrix_y[row][0]
                        n1 = predictedData[row][0]
                        if (isClassification == False):
                            if (((n1*n2) != 0)):
                                newAcurracyValueToAdd = (1-(abs(n2-n1)/abs(n2)))
                                if (newAcurracyValueToAdd < 0):
                                    newAcurracyValueToAdd = 0
                                predictionAcurracy = predictionAcurracy + newAcurracyValueToAdd
                        if (isClassification == True):
                            if (abs(n1) > abs(n2)): # n2 has to be the one with the highest value with respect to n1
                                n2 = predictedData[row][0]
                                n1 = matrix_y[row][0]
                            if ((n1==0) and (n2>=-1 and n2<=1) and (n2!=0)):
                                predictionAcurracy = predictionAcurracy + ((1-abs(n2))/(1-n1))
                            if (n1==n2):
                                predictionAcurracy = predictionAcurracy + 1
                    predictionAcurracy = predictionAcurracy/numberOfDataPoints*100
                    temporalRow = []
                    temporalRow.append(predictionAcurracy)
                    temporalRow.append(currentMatrix_b)
                    temporalRow.append(currentMatrix_x)
                    allAccuracies.append(temporalRow)
        
                    # We save the current the modeling results if they are better than
                    # the actual best
                    currentBestAccuracy = bestModelingResults[1]
                    if (predictionAcurracy > currentBestAccuracy):
                        bestModelingResults = []
                        bestModelingResults.append(currentMatrix_b)
                        bestModelingResults.append(predictionAcurracy)
                        bestModelingResults.append(predictedData)
                        bestModelingResults.append("Coefficients distribution is as follows: y = bo + b1*x1 + b2*x1^2 + b3*x2 + b4*x2^2 + b5*x1*x2")
        # Alongside the information of the best model obtained, we add the
        # modeled information of ALL the models obtained to the variable that
        # we will return in this method
        bestModelingResults.append(allAccuracies)
        return bestModelingResults
        
    
    """
    getCustomizedMultipleThirdOrderPolynomialRegression(evtfbmip="True to indicate to Eliminate Variables To Find Better Model If Possible. False if the contrary is desired",
                                                        isClassification="set to True if you are solving a classification problem. False if otherwise")
    
    This method obtains the best solution of a customized 3rd order model when
    using specifically 2 independent variables and were the equation to solve
    is the following:
    y = bo + b1*x1 + b2*x1^2 + b3*x1^3 + b4*x2 + b5*x2^2 + b6*x2^3 + b7*x1*x2 + b8*x1^2*x2 + b9*x1*x2^2
    IMPORTANT NOTE: The same base algorithm used in the method
    "getCustomizedMultipleSecondOrderPolynomialRegression()" was applied in
    this one. This is important to mention because the algorithm i created in
    that method demonstrated to be superior of that one used in the book
    "Probabilidad y estadistica para ingenieria & ciencias (Walpole, Myers,
    Myers, Ye)". See that method's description to see more information about
    this.
    
    CODE EXAMPLE:
        matrix_y = [
                [14.05],
                [10.55],
                [7.55],
                [14.93],
                [9.48],
                [6.59],
                [16.56],
                [13.63],
                [9.23],
                [15.85],
                [11.75],
                [8.78],
                [22.41],
                [18.55],
                [15.93],
                [21.66],
                [17.98],
                [16.44]
                ]
        matrix_x = [
                [75, 15],
                [100, 15],
                [125, 15],
                [75, 17.5],
                [100, 17.5],
                [125, 17.5],
                [75, 20],
                [100, 20],
                [125, 20],
                [75, 22.5],
                [100, 22.5],
                [125, 22.5],
                [75, 25],
                [100, 25],
                [125, 25],
                [75, 27.5],
                [100, 27.5],
                [125, 27.5]
                ]
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        regression = mSL.Regression(matrix_x, matrix_y)
        modelingResults = regression.getCustomizedMultipleThirdOrderPolynomialRegression(evtfbmip=True, isClassification=False)
        modelCoefficients = modelingResults[0]
        acurracy = modelingResults[1]
        predictedData = modelingResults[2]
        coefficientDistribution = modelingResults[3]
        allModeledAccuracies = modelingResults[4]
        
    RESULT OF CODE:
        modelCoefficients =
        [[118.62284443469252],
         [2.6850685669390923e-10],
         [0],
         [2.711111111130216e-06],
         [-14.043715503707062],
         [0.7156842175145357],
         [-0.011482404265578339],
         [-0.024609341568850862],
         [0],
         [0.0006459332618172914]]
        
        acurracy =
        92.07595419629946
        
        predictedData =
        [[14.601310971885873],
         [10.5735435991239],
         [7.56244289303574],
         [14.177873191206809],
         [9.924073908458023],
         [6.686941292383061],
         [15.770722763127356],
         [11.492745714709685],
         [8.23143533296583],
         [18.303384287749555],
         [14.203083617980887],
         [11.11944961488603],
         [20.699382365175477],
         [16.978612218373712],
         [14.274508738245757],
         [21.882241595507075],
         [18.742856115990087],
         [16.62013730314699]]
        
        coefficientDistribution =
        'Coefficients distribution is as follows: y = bo + b1*x1 + b2*x1^2 + b3*x1^3 + b4*x2 + b5*x2^2 + b6*x2^3 + b7*x1*x2 + b8*x1^2*x2 + b9*x1*x2^2'
        
        allModeledAccuracies["independent variable distribution used to get a model"]["model accuracy", "model coefficients obtained but with original distribution", "matrix x data"] =
        # NOTE: since this variable contains large amounts of information, it
        #       will not be displayed but only described on how to use it.
    """
    def getCustomizedMultipleThirdOrderPolynomialRegression(self, evtfbmip=False, isClassification=True):
        # We import the libraries we want to use and we create the class we
        # use from it
        from ..linearAlgebra import MortrackLinearAlgebraLibrary as mLAL
        matrixMath = mLAL.MatrixMath()
        # We define the variables to use within our code
        matrix_y =self.y_samplesList
        rowLengthOfBothMatrixes = len(matrix_y)
        x1 = 0
        x2 = 1
        # ----- WE GET THE FIRST MODEL EVALUATION RESULTS ----- #
        # MATRIX X MATHEMATICAL PROCEDURE to create a matrix that contains
        # the x values of the following equation we want to solve and in that
        # same variable formation:
        # y = bo + b1*x1 + b2*x1^2 + b3*x1^3 + b4*x2 + b5*x2^2 + b6*x2^3 + b7*x1*x2 + b8*x1^2*x2 + b9*x1*x2^2
        currentMatrix_x = []
        for row in range(0, rowLengthOfBothMatrixes):
            temporalRow = []
            temporalRow.append(1)
            temporalRow.append(self.x_samplesList[row][x1])
            temporalRow.append(self.x_samplesList[row][x1]**2)
            temporalRow.append(self.x_samplesList[row][x1]**3)
            temporalRow.append(self.x_samplesList[row][x2])
            temporalRow.append(self.x_samplesList[row][x2]**2)
            temporalRow.append(self.x_samplesList[row][x2]**3)
            temporalRow.append(self.x_samplesList[row][x1]*self.x_samplesList[row][x2])
            temporalRow.append((self.x_samplesList[row][x1]**2)*self.x_samplesList[row][x2])
            temporalRow.append(self.x_samplesList[row][x1]*(self.x_samplesList[row][x2]**2))
            currentMatrix_x.append(temporalRow)
        originalMatrix_x = currentMatrix_x
            
        # We get the Transposed matrix of matrix X. NOTE: We create a temporal
        # variable to save matrix x because remember that in python, children
        # and parent inheritance is ignored when using clases
        temporalMatrix1 = currentMatrix_x
        transposedMatrix_X = matrixMath.getTransposedMatrix(temporalMatrix1)
        # WE GET MATRIX A.  NOTE: We create a temporal
        # variable to save matrix x because remember that in python, children
        # and parent inheritance is ignored when using clases
        temporalMatrix1 = transposedMatrix_X
        temporalMatrix2 = currentMatrix_x
        matrix_A = matrixMath.getMultiplication(temporalMatrix1, temporalMatrix2)
        # WE GET MATRIX g. NOTE: We create a temporal
        # variable to save matrix x because remember that in python, children
        # and parent inheritance is ignored when using clases
        temporalMatrix1 = transposedMatrix_X
        temporalMatrix2 = matrix_y
        matrix_g = matrixMath.getMultiplication(temporalMatrix1, temporalMatrix2)
        # We get inverse matrix of matrix A.
        inversedMatrix_A = matrixMath.getInverse(matrix_A)
        # We get matrix b, which will contain the coeficient values
        matrix_b = matrixMath.getMultiplication(inversedMatrix_A, matrix_g)
        
        # We determine the accuracy of the obtained coefficients
        predictedData = []
        for row in range(0, len(matrix_y)):
            temporalRow = []
            actualIc = matrix_b[0][0] + matrix_b[1][0]*self.x_samplesList[row][x1] + matrix_b[2][0]*self.x_samplesList[row][x1]**2 + matrix_b[3][0]*self.x_samplesList[row][x1]**3 + matrix_b[4][0]*self.x_samplesList[row][x2] + matrix_b[5][0]*self.x_samplesList[row][x2]**2 + matrix_b[6][0]*self.x_samplesList[row][x2]**3 + matrix_b[7][0]*self.x_samplesList[row][x1]*self.x_samplesList[row][x2] + matrix_b[8][0]*(self.x_samplesList[row][x1]**2)*self.x_samplesList[row][x2] + matrix_b[9][0]*self.x_samplesList[row][x1]*(self.x_samplesList[row][x2]**2)
            temporalRow.append(actualIc)
            predictedData.append(temporalRow)
        predictionAcurracy = 0
        numberOfDataPoints = len(matrix_y)
        for row in range(0, numberOfDataPoints):
            n2 = matrix_y[row][0]
            n1 = predictedData[row][0]
            if (isClassification == False):
                if (((n1*n2) != 0)):
                    newAcurracyValueToAdd = (1-(abs(n2-n1)/abs(n2)))
                    if (newAcurracyValueToAdd < 0):
                        newAcurracyValueToAdd = 0
                    predictionAcurracy = predictionAcurracy + newAcurracyValueToAdd
            if (isClassification == True):
                if (abs(n1) > abs(n2)): # n2 has to be the one with the highest value with respect to n1
                    n2 = predictedData[row][0]
                    n1 = matrix_y[row][0]
                if ((n1==0) and (n2>=-1 and n2<=1) and (n2!=0)):
                    predictionAcurracy = predictionAcurracy + ((1-abs(n2))/(1-n1))
                if (n1==n2):
                    predictionAcurracy = predictionAcurracy + 1
        predictionAcurracy = predictionAcurracy/numberOfDataPoints*100
        
        # We save the current the modeling results
        bestModelingResults = []
        bestModelingResults.append(matrix_b)
        bestModelingResults.append(predictionAcurracy)
        bestModelingResults.append(predictedData)
        bestModelingResults.append("Coefficients distribution is as follows: y = bo + b1*x1 + b2*x1^2 + b3*x1^3 + b4*x2 + b5*x2^2 + b6*x2^3 + b7*x1*x2 + b8*x1^2*x2 + b9*x1*x2^2")
        allAccuracies = []
        temporalRow = []
        temporalRow.append(bestModelingResults[1])
        temporalRow.append(bestModelingResults[0])
        temporalRow.append(originalMatrix_x)
        allAccuracies.append(temporalRow)
        
        # ----- WE START SEARCHING FOR THE BEST MODELING RESULTS ----- #
        # We define a variable to save the search patterns in original matrix x
        from .MortrackML_Library import Combinations
        possibleCombinations = []
        for n in range (0, len(originalMatrix_x[0])):
            possibleCombinations.append(n)
        combinations = Combinations(possibleCombinations)
        searchPatterns = combinations.getPositionCombinationsList()
        searchPatterns.pop(0) # We remove the first one because we already did it
        # We start to search for the coefficients that give us the best accuracy
        for currentSearchPattern in range(0, len(searchPatterns)):
            currentMatrix_x = [ [ 0 for i in range(len(originalMatrix_x[0])) ] for j in range(rowLengthOfBothMatrixes) ]
            # We assign the current distribution that we want to study of the
            # variables of the matrix x, to evaluate its resulting regression
            # coefficients
            for currentColumnOfMatrix_x in range(0, len(originalMatrix_x[0])):
                if (searchPatterns[currentSearchPattern][currentColumnOfMatrix_x] == 0):
                    for row in range(0, rowLengthOfBothMatrixes):
                        currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][0]
                if (searchPatterns[currentSearchPattern][currentColumnOfMatrix_x] == 1):
                    for row in range(0, rowLengthOfBothMatrixes):
                        currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][1]
                if (searchPatterns[currentSearchPattern][currentColumnOfMatrix_x] == 2):
                    for row in range(0, rowLengthOfBothMatrixes):
                        currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][2]
                if (searchPatterns[currentSearchPattern][currentColumnOfMatrix_x] == 3):
                    for row in range(0, rowLengthOfBothMatrixes):
                        currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][3]
                if (searchPatterns[currentSearchPattern][currentColumnOfMatrix_x] == 4):
                    for row in range(0, rowLengthOfBothMatrixes):
                        currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][4]
                if (searchPatterns[currentSearchPattern][currentColumnOfMatrix_x] == 5):
                    for row in range(0, rowLengthOfBothMatrixes):
                        currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][5]
                if (searchPatterns[currentSearchPattern][currentColumnOfMatrix_x] == 6):
                    for row in range(0, rowLengthOfBothMatrixes):
                        currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][6]
                if (searchPatterns[currentSearchPattern][currentColumnOfMatrix_x] == 7):
                    for row in range(0, rowLengthOfBothMatrixes):
                        currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][7]
                if (searchPatterns[currentSearchPattern][currentColumnOfMatrix_x] == 8):
                    for row in range(0, rowLengthOfBothMatrixes):
                        currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][8]
                if (searchPatterns[currentSearchPattern][currentColumnOfMatrix_x] == 9):
                    for row in range(0, rowLengthOfBothMatrixes):
                        currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][9]
                        
            # We get the Transposed matrix of matrix X. NOTE: We create a temporal
            # variable to save matrix x because remember that in python, children
            # and parent inheritance is ignored when using clases
            temporalMatrix1 = currentMatrix_x
            transposedMatrix_X = matrixMath.getTransposedMatrix(temporalMatrix1)
            # WE GET MATRIX A.  NOTE: We create a temporal
            # variable to save matrix x because remember that in python, children
            # and parent inheritance is ignored when using clases
            temporalMatrix1 = transposedMatrix_X
            temporalMatrix2 = currentMatrix_x
            matrix_A = matrixMath.getMultiplication(temporalMatrix1, temporalMatrix2)
            # WE GET MATRIX g. NOTE: We create a temporal
            # variable to save matrix x because remember that in python, children
            # and parent inheritance is ignored when using clases
            temporalMatrix1 = transposedMatrix_X
            temporalMatrix2 = matrix_y
            matrix_g = matrixMath.getMultiplication(temporalMatrix1, temporalMatrix2)
            # We get inverse matrix of matrix A.
            inversedMatrix_A = matrixMath.getInverse(matrix_A)
            # We get matrix b, which will contain the coeficient values
            matrix_b = matrixMath.getMultiplication(inversedMatrix_A, matrix_g)
            
            # ----- WE DETERMINE THE ACCURACY OF THE OBTAINED COEFFICIENTS ----- #
            # We re-arrange the obtained coefficients to then evaluate this
            # model
            currentMatrix_b = [ [ 0 for i in range(1) ] for j in range(len(originalMatrix_x[0])) ]
            for row in range(0, len(originalMatrix_x[0])):
                trueRowOfCoefficient = searchPatterns[currentSearchPattern][row]
                currentMatrix_b[trueRowOfCoefficient][0] = matrix_b[row][0]
            # We obtain the predicted data through the current obtained
            # coefficients
            predictedData = []
            for row in range(0, len(matrix_y)):
                temporalRow = []
                actualIc = currentMatrix_b[0][0] + currentMatrix_b[1][0]*self.x_samplesList[row][x1] + currentMatrix_b[2][0]*self.x_samplesList[row][x1]**2 + currentMatrix_b[3][0]*self.x_samplesList[row][x1]**3 + currentMatrix_b[4][0]*self.x_samplesList[row][x2] + currentMatrix_b[5][0]*self.x_samplesList[row][x2]**2 + currentMatrix_b[6][0]*self.x_samplesList[row][x2]**3 + currentMatrix_b[7][0]*self.x_samplesList[row][x1]*self.x_samplesList[row][x2] + currentMatrix_b[8][0]*(self.x_samplesList[row][x1]**2)*self.x_samplesList[row][x2] + currentMatrix_b[9][0]*self.x_samplesList[row][x1]*(self.x_samplesList[row][x2]**2)
                temporalRow.append(actualIc)
                predictedData.append(temporalRow)
            
            predictionAcurracy = 0
            numberOfDataPoints = len(matrix_y)
            for row in range(0, numberOfDataPoints):
                n2 = matrix_y[row][0]
                n1 = predictedData[row][0]
                if (isClassification == False):
                    if (((n1*n2) != 0)):
                        newAcurracyValueToAdd = (1-(abs(n2-n1)/abs(n2)))
                        if (newAcurracyValueToAdd < 0):
                            newAcurracyValueToAdd = 0
                        predictionAcurracy = predictionAcurracy + newAcurracyValueToAdd
                if (isClassification == True):
                    if (abs(n1) > abs(n2)): # n2 has to be the one with the highest value with respect to n1
                        n2 = predictedData[row][0]
                        n1 = matrix_y[row][0]
                    if ((n1==0) and (n2>=-1 and n2<=1) and (n2!=0)):
                        predictionAcurracy = predictionAcurracy + ((1-abs(n2))/(1-n1))
                    if (n1==n2):
                        predictionAcurracy = predictionAcurracy + 1
            predictionAcurracy = predictionAcurracy/numberOfDataPoints*100
            temporalRow = []
            temporalRow.append(predictionAcurracy)
            temporalRow.append(currentMatrix_b)
            temporalRow.append(currentMatrix_x)
            allAccuracies.append(temporalRow)

            # We save the current the modeling results if they are better than
            # the actual best
            currentBestAccuracy = bestModelingResults[1]
            if (predictionAcurracy > currentBestAccuracy):
                bestModelingResults = []
                bestModelingResults.append(currentMatrix_b)
                bestModelingResults.append(predictionAcurracy)
                bestModelingResults.append(predictedData)
                bestModelingResults.append("Coefficients distribution is as follows: y = bo + b1*x1 + b2*x1^2 + b3*x1^3 + b4*x2 + b5*x2^2 + b6*x2^3 + b7*x1*x2 + b8*x1^2*x2 + b9*x1*x2^2")
        
        if (evtfbmip == True):
            # ----------------------------------------------------------------------------------------------- #
            # ----- We now get all possible combinations/permutations with the elements of our equation ----- #
            # ----------------------------------------------------------------------------------------------- #
            customizedPermutations = combinations.getCustomizedPermutationList()
            customizedPermutations.pop(0) # We remove the null value
            customizedPermutations.pop(len(customizedPermutations)-1) # We remove the last one because we already did it
            for actualPermutation in range(0, len(customizedPermutations)):
                newOriginalMatrix_x = []
                for row in range(0, rowLengthOfBothMatrixes):
                    temporalRow = []
                    for column in range(0, len(customizedPermutations[actualPermutation])):
                        temporalRow.append(originalMatrix_x[row][customizedPermutations[actualPermutation][column]])
                    newOriginalMatrix_x.append(temporalRow)
                
            # ----- WE START SEARCHING FOR THE BEST MODELING RESULTS USING CURRENT PERMUTATION ----- #
                # We define a variable to save the search patterns in original matrix x
                possibleCombinations = []
                for n in range (0, len(newOriginalMatrix_x[0])):
                    possibleCombinations.append(n)
                combinations = Combinations(possibleCombinations)
                searchPatterns = combinations.getPositionCombinationsList()
                
                # We start to search for the coefficients that give us the best accuracy
                for currentSearchPattern in range(0, len(searchPatterns)):
                    currentMatrix_x = [ [ 0 for i in range(len(newOriginalMatrix_x[0])) ] for j in range(rowLengthOfBothMatrixes) ]
                    # We assign the current distribution that we want to study of the
                    # variables of the matrix x, to evaluate its resulting regression
                    # coefficients
                    for currentColumnOfMatrix_x in range(0, len(newOriginalMatrix_x[0])):    
                        if (customizedPermutations[actualPermutation][searchPatterns[currentSearchPattern][currentColumnOfMatrix_x]] == 0):
                            for row in range(0, rowLengthOfBothMatrixes):
                                currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][0]
                        if (customizedPermutations[actualPermutation][searchPatterns[currentSearchPattern][currentColumnOfMatrix_x]] == 1):
                            for row in range(0, rowLengthOfBothMatrixes):
                                currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][1]
                        if (customizedPermutations[actualPermutation][searchPatterns[currentSearchPattern][currentColumnOfMatrix_x]] == 2):
                            for row in range(0, rowLengthOfBothMatrixes):
                                currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][2]
                        if (customizedPermutations[actualPermutation][searchPatterns[currentSearchPattern][currentColumnOfMatrix_x]] == 3):
                            for row in range(0, rowLengthOfBothMatrixes):
                                currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][3]
                        if (customizedPermutations[actualPermutation][searchPatterns[currentSearchPattern][currentColumnOfMatrix_x]] == 4):
                            for row in range(0, rowLengthOfBothMatrixes):
                                currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][4]
                        if (customizedPermutations[actualPermutation][searchPatterns[currentSearchPattern][currentColumnOfMatrix_x]] == 5):
                            for row in range(0, rowLengthOfBothMatrixes):
                                currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][5]
                        if (customizedPermutations[actualPermutation][searchPatterns[currentSearchPattern][currentColumnOfMatrix_x]] == 6):
                            for row in range(0, rowLengthOfBothMatrixes):
                                currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][6]
                        if (customizedPermutations[actualPermutation][searchPatterns[currentSearchPattern][currentColumnOfMatrix_x]] == 7):
                            for row in range(0, rowLengthOfBothMatrixes):
                                currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][7]
                        if (customizedPermutations[actualPermutation][searchPatterns[currentSearchPattern][currentColumnOfMatrix_x]] == 8):
                            for row in range(0, rowLengthOfBothMatrixes):
                                currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][8]
                        if (customizedPermutations[actualPermutation][searchPatterns[currentSearchPattern][currentColumnOfMatrix_x]] == 9):
                            for row in range(0, rowLengthOfBothMatrixes):
                                currentMatrix_x[row][currentColumnOfMatrix_x] = originalMatrix_x[row][9]
                    # We get the Transposed matrix of matrix X. NOTE: We create a temporal
                    # variable to save matrix x because remember that in python, children
                    # and parent inheritance is ignored when using clases
                    temporalMatrix1 = currentMatrix_x
                    transposedMatrix_X = matrixMath.getTransposedMatrix(temporalMatrix1)
                    # WE GET MATRIX A.  NOTE: We create a temporal
                    # variable to save matrix x because remember that in python, children
                    # and parent inheritance is ignored when using clases
                    temporalMatrix1 = transposedMatrix_X
                    temporalMatrix2 = currentMatrix_x
                    matrix_A = matrixMath.getMultiplication(temporalMatrix1, temporalMatrix2)
                    # WE GET MATRIX g. NOTE: We create a temporal
                    # variable to save matrix x because remember that in python, children
                    # and parent inheritance is ignored when using clases
                    temporalMatrix1 = transposedMatrix_X
                    temporalMatrix2 = matrix_y
                    matrix_g = matrixMath.getMultiplication(temporalMatrix1, temporalMatrix2)
                    # We get inverse matrix of matrix A.
                    inversedMatrix_A = matrixMath.getInverse(matrix_A)
                    # We get matrix b, which will contain the coeficient values
                    matrix_b = matrixMath.getMultiplication(inversedMatrix_A, matrix_g)
                    
                    # ----- WE DETERMINE THE ACCURACY OF THE OBTAINED COEFFICIENTS ----- #
                    # We re-arrange the obtained coefficients to then evaluate this
                    # model
                    currentMatrix_b = [ [ 0 for i in range(1) ] for j in range(len(originalMatrix_x[0])) ]
                    for row in range(0, len(newOriginalMatrix_x[0])):
                        trueRowOfCoefficient = customizedPermutations[actualPermutation][searchPatterns[currentSearchPattern][row]]
                        currentMatrix_b[trueRowOfCoefficient][0] = matrix_b[row][0]
                    # We obtain the predicted data through the current obtained
                    # coefficients
                    predictedData = []
                    for row in range(0, len(matrix_y)):
                        temporalRow = []
                        actualIc = currentMatrix_b[0][0] + currentMatrix_b[1][0]*self.x_samplesList[row][x1] + currentMatrix_b[2][0]*self.x_samplesList[row][x1]**2 + currentMatrix_b[3][0]*self.x_samplesList[row][x1]**3 + currentMatrix_b[4][0]*self.x_samplesList[row][x2] + currentMatrix_b[5][0]*self.x_samplesList[row][x2]**2 + currentMatrix_b[6][0]*self.x_samplesList[row][x2]**3 + currentMatrix_b[7][0]*self.x_samplesList[row][x1]*self.x_samplesList[row][x2] + currentMatrix_b[8][0]*(self.x_samplesList[row][x1]**2)*self.x_samplesList[row][x2] + currentMatrix_b[9][0]*self.x_samplesList[row][x1]*(self.x_samplesList[row][x2]**2)
                        temporalRow.append(actualIc)
                        predictedData.append(temporalRow)
                    
                    predictionAcurracy = 0
                    numberOfDataPoints = len(matrix_y)
                    for row in range(0, numberOfDataPoints):
                        n2 = matrix_y[row][0]
                        n1 = predictedData[row][0]
                        if (isClassification == False):
                            if (((n1*n2) != 0)):
                                newAcurracyValueToAdd = (1-(abs(n2-n1)/abs(n2)))
                                if (newAcurracyValueToAdd < 0):
                                    newAcurracyValueToAdd = 0
                                predictionAcurracy = predictionAcurracy + newAcurracyValueToAdd
                        if (isClassification == True):
                            if (abs(n1) > abs(n2)): # n2 has to be the one with the highest value with respect to n1
                                n2 = predictedData[row][0]
                                n1 = matrix_y[row][0]
                            if ((n1==0) and (n2>=-1 and n2<=1) and (n2!=0)):
                                predictionAcurracy = predictionAcurracy + ((1-abs(n2))/(1-n1))
                            if (n1==n2):
                                predictionAcurracy = predictionAcurracy + 1
                    predictionAcurracy = predictionAcurracy/numberOfDataPoints*100
                    temporalRow = []
                    temporalRow.append(predictionAcurracy)
                    temporalRow.append(currentMatrix_b)
                    temporalRow.append(currentMatrix_x)
                    allAccuracies.append(temporalRow)
        
                    # We save the current the modeling results if they are better than
                    # the actual best
                    currentBestAccuracy = bestModelingResults[1]
                    if (predictionAcurracy > currentBestAccuracy):
                        bestModelingResults = []
                        bestModelingResults.append(currentMatrix_b)
                        bestModelingResults.append(predictionAcurracy)
                        bestModelingResults.append(predictedData)
                        bestModelingResults.append("Coefficients distribution is as follows: y = bo + b1*x1 + b2*x1^2 + b3*x1^3 + b4*x2 + b5*x2^2 + b6*x2^3 + b7*x1*x2 + b8*x1^2*x2 + b9*x1*x2^2")
        # Alongside the information of the best model obtained, we add the
        # modeled information of ALL the models obtained to the variable that
        # we will return in this method
        bestModelingResults.append(allAccuracies)
        return bestModelingResults
        
    """
    predictLinearLogisticRegression(coefficients="We give the Linear Logistic mathematical coefficients that we want to predict with")
    
    This method returns the predicting values of the independent input values
    that you assign in the local variable of this class: "self.x_samplesList".
    The prediction will be made accordingly to the coefficients and
    configuration specified in the arguments of this method.
    
    CODE EXAMPLE:
        matrix_x = [
                [0,2],
                [1,3],
                [2,4],
                [3,5],
                [4,6],
                [5,7],
                [6,8],
                [7,9],
                [8,10],
                [9,11]
                ]
        
        matrix_y = [
                [0],
                [0],
                [1],
                [0],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1]
                ]
        
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        regression = mSL.Regression(matrix_x, matrix_y)
        # evtfbmip stands for "Eliminate Variables To Find Better Model If Possible"
        modelingResults = regression.getLinearLogisticRegression(evtfbmip=True)
        modelCoefficients = modelingResults[0]
        acurracy = modelingResults[1]
        predictedData = modelingResults[2]
        coefficientDistribution = modelingResults[3]
        allModeledAccuracies = modelingResults[4]
        
        # ------------------------------------- #
        # ----- WE VISUALIZE OUR RESULTS  ----- #
        # ------------------------------------- #
        predictThisValues = [
                [0,1],
                [4,4],
                [6,6],
                [10,10],
                [1,8]
                ]
        regression.set_xSamplesList(predictThisValues)
        predictedValues = regression.predictLinearLogisticRegression(coefficients=modelCoefficients)
        
        
    EXPECTED CODE RESULT:
        predictedValues =
        [[0.5],
         [0.999978721536189],
         [0.9999991162466249],
         [0.9999999984756125],
         [1.7295081461872963e-11]]
    """
    def predictLinearLogisticRegression(self, coefficients):
        import math
        numberOfRows = len(self.x_samplesList)
        # We determine the accuracy of the obtained coefficientsfor the 
        # Probability Equation of the Logistic Regression Equation
        predictedData = []
        numberOfIndependentVariables = len(self.x_samplesList[0])
        for row in range(0, numberOfRows):
            temporalRow = []
            actualIc = coefficients[0][0]
            for currentIndependentVariable in range(0, numberOfIndependentVariables):
                actualIc = actualIc + coefficients[currentIndependentVariable+1][0]*self.x_samplesList[row][currentIndependentVariable]
            actualIc = math.exp(actualIc)
            actualIc = actualIc/(1+actualIc)
            temporalRow.append(actualIc)
            predictedData.append(temporalRow)
        
        # We return the predicted data
        return predictedData
    
    """
    predictGaussianRegression(coefficients="We give the Linear Logistic mathematical coefficients that we want to predict with")
    
    This method returns the predicting values of the independent input values
    that you assign in the local variable of this class: "self.x_samplesList".
    The prediction will be made accordingly to the coefficients and
    configuration specified in the arguments of this method.
    
    CODE EXAMPLE:
        # We will simulate a dataset that you would normally have in its original form
        matrix_x = [
             [2, 3],
             [3, 2],
             [4, 3],
             [3, 4],
             [1, 3],
             [3, 1],
             [5, 3],
             [3, 5]
             ]
        matrix_y = [
             [1],
             [1],
             [1],
             [1],
             [0],
             [0],
             [0],
             [0]
             ]
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        regression = mSL.Regression(matrix_x, matrix_y)
        modelingResults = regression.getGaussianRegression()
        modelCoefficients = modelingResults[0]
        acurracy = modelingResults[1]
        predictedData = modelingResults[2]
        coefficientDistribution = modelingResults[3]
        allModeledAccuracies = modelingResults[4]
        
        # ------------------------------------- #
        # ----- WE VISUALIZE OUR RESULTS  ----- #
        # ------------------------------------- #
        predictThisValues = [
                [0,1],
                [4,4],
                [6,6],
                [10,10],
                [1,8]
                ]
        regression.set_xSamplesList(predictThisValues)
        predictedValues = regression.predictGaussianRegression(coefficients=modelCoefficients)
        
        
    EXPECTED CODE RESULT:
        predictedValues =
        [[1.003006010014743e-12],
         [0.09993332221727314],
         [1.0046799183277663e-17],
         [1.0318455659367212e-97],
         [1.0083723565531913e-28]]
    """
    def predictGaussianRegression(self, coefficients):
        import math
        numberOfRows = len(self.x_samplesList)
        # We obtain the predicted data of the desired independent given values
        predictedData = []
        orderOfThePolynomial = 2
        numberOfIndependentVariables = (len(coefficients)-1)
        for row in range(0, numberOfRows):
            temporalRow = []
            actualIc = coefficients[0][0]
            currentOrderOfThePolynomial = 1
            currentVariable = 0
            for currentIndependentVariable in range(0, numberOfIndependentVariables):
                if (currentOrderOfThePolynomial == (orderOfThePolynomial+1)):
                    currentOrderOfThePolynomial = 1
                    currentVariable = currentVariable + 1
                actualIc = actualIc + coefficients[currentIndependentVariable+1][0]*self.x_samplesList[row][currentVariable]**(currentOrderOfThePolynomial)
                currentOrderOfThePolynomial = currentOrderOfThePolynomial + 1
            temporalRow.append(math.exp(-(actualIc)))
            predictedData.append(temporalRow)
                    
        # We return the predicted data
        return predictedData
    
    """
    predictLinearRegression(coefficients="We give the Linear Logistic mathematical coefficients that we want to predict with")
    
    This method returns the predicting values of the independent input values
    that you assign in the local variable of this class: "self.x_samplesList".
    The prediction will be made accordingly to the coefficients and
    configuration specified in the arguments of this method.
    
    CODE EXAMPLE:
        matrix_x = [
                [0],
                [1],
                [2],
                [3],
                [4],
                [5],
                [6],
                [7],
                [8],
                [9]
                ]
        matrix_y = [
                [8.5],
                [9.7],
                [10.7],
                [11.5],
                [12.1],
                [14],
                [13.3],
                [16.2],
                [17.3],
                [17.5]
                ]
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        regression = mSL.Regression(matrix_x, matrix_y)
        modelingResults = regression.getLinearRegression(isClassification=False)
        modelCoefficients = modelingResults[0]
        acurracy = modelingResults[1]
        predictedData = modelingResults[2]
        coefficientDistribution = modelingResults[3]
        allModeledAccuracies = modelingResults[4]
        
        # ------------------------------------- #
        # ----- WE VISUALIZE OUR RESULTS  ----- #
        # ------------------------------------- #
        predictThisValues = [
                [0],
                [4],
                [6],
                [10],
                [1]
                ]
        regression.set_xSamplesList(predictThisValues)
        predictedValues = regression.predictLinearRegression(coefficients=modelCoefficients)
        
        
    EXPECTED CODE RESULT:
        predictedValues =
        [[8.470909090909096],
         [12.56787878787879],
         [14.616363636363639],
         [18.71333333333333],
         [9.49515151515152]]
    """
    def predictLinearRegression(self, coefficients):
        numberOfRows = len(self.x_samplesList)
        # We obtain the predicted data of the desired independent given values
        predictedData = []
        for row in range(0, numberOfRows):
            temporalRow = []
            actualIc = coefficients[0][0] + coefficients[1][0]*self.x_samplesList[row][0]
            temporalRow.append(actualIc)
            predictedData.append(temporalRow)
        # We return the predicted data
        return predictedData
    
    """
    predictMultipleLinearRegression(coefficients="We give the Linear Logistic mathematical coefficients that we want to predict with")
    
    This method returns the predicting values of the independent input values
    that you assign in the local variable of this class: "self.x_samplesList".
    The prediction will be made accordingly to the coefficients and
    configuration specified in the arguments of this method.
    
    CODE EXAMPLE:
        # matrix_y = [expectedResult]
        matrix_y = [
                [25.5],
                [31.2],
                [25.9],
                [38.4],
                [18.4],
                [26.7],
                [26.4],
                [25.9],
                [32],
                [25.2],
                [39.7],
                [35.7],
                [26.5]
                ]
        # matrix_x = [variable1, variable2, variable3]
        matrix_x = [
                [1.74, 5.3, 10.8],
                [6.32, 5.42, 9.4],
                [6.22, 8.41, 7.2],
                [10.52, 4.63, 8.5],
                [1.19, 11.6, 9.4],
                [1.22, 5.85, 9.9],
                [4.1, 6.62, 8],
                [6.32, 8.72, 9.1],
                [4.08, 4.42, 8.7],
                [4.15, 7.6, 9.2],
                [10.15, 4.83, 9.4],
                [1.72, 3.12, 7.6],
                [1.7, 5.3, 8.2]
                ]
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        regression = mSL.Regression(matrix_x, matrix_y)
        # "evtfbmip" stands for "Eliminate Variables To Find Better Model If Possible"
        modelingResults = regression.getMultipleLinearRegression(evtfbmip = True, isClassification=False)
        modelCoefficients = modelingResults[0]
        acurracy = modelingResults[1]
        predictedData = modelingResults[2]
        coefficientDistribution = modelingResults[3]
        allModeledAccuracies = modelingResults[4]
        
        # ------------------------------------- #
        # ----- WE VISUALIZE OUR RESULTS  ----- #
        # ------------------------------------- #
        predictThisValues = [
                [0,1,1],
                [4,4,4],
                [6,6,6],
                [10,10,10],
                [1,8,9]
                ]
        regression.set_xSamplesList(predictThisValues)
        predictedValues = regression.predictMultipleLinearRegression(coefficients=modelCoefficients)
        
        
    EXPECTED CODE RESULT:
        predictedValues =
        [[34.22503543093558],
         [32.73815713171364],
         [31.059896530994866],
         [27.703375329557314],
         [22.168047717282477]]
    """
    def predictMultipleLinearRegression(self, coefficients):
        numberOfRows = len(self.x_samplesList)
        # We obtain the predicted data of the desired independent given values
        predictedData = []
        numberOfIndependentVariables = len(self.x_samplesList[0])
        for row in range(0, numberOfRows):
            temporalRow = []
            actualIc = coefficients[0][0]
            for currentIndependentVariable in range(0, numberOfIndependentVariables):
                actualIc = actualIc + coefficients[currentIndependentVariable+1][0]*self.x_samplesList[row][currentIndependentVariable]
            temporalRow.append(actualIc)
            predictedData.append(temporalRow)
        # We return the predicted data
        return predictedData
    
    """
    predictPolynomialRegression(coefficients="We give the Linear Logistic mathematical coefficients that we want to predict with")
    
    This method returns the predicting values of the independent input values
    that you assign in the local variable of this class: "self.x_samplesList".
    The prediction will be made accordingly to the coefficients and
    configuration specified in the arguments of this method.
    
    CODE EXAMPLE:
        matrix_y = [
                [3.4769e-11],
                [7.19967e-11],
                [1.59797e-10],
                [3.79298e-10]
                ]
        matrix_x = [
                [-0.7],
                [-0.65],
                [-0.6],
                [-0.55]
                ]
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        regression = mSL.Regression(matrix_x, matrix_y)
        # "orderOfThePolynomial" = "whole number to represent the desired order of the polynomial model to find"
        # "evtfbmip" stands for "Eliminate Variables To Find Better Model If Possible"
        modelingResults = regression.getPolynomialRegression(orderOfThePolynomial=3, evtfbmip=True, isClassification=False)
        modelCoefficients = modelingResults[0]
        acurracy = modelingResults[1]
        predictedData = modelingResults[2]
        coefficientDistribution = modelingResults[3]
        allModeledAccuracies = modelingResults[4]
        
        # ------------------------------------- #
        # ----- WE VISUALIZE OUR RESULTS  ----- #
        # ------------------------------------- #
        predictThisValues = [
                [0],
                [4],
                [6],
                [10],
                [1]
                ]
        regression.set_xSamplesList(predictThisValues)
        predictedValues = regression.predictPolynomialRegression(coefficients=modelCoefficients)
        
        
    EXPECTED CODE RESULT:
        predictedValues =
        [[3.468869185343018e-08],
         [1.1099322065704926e-05],
         [3.226470574414124e-05],
         [0.000131822599137008],
         [5.151422907494728e-07]]
    """
    def predictPolynomialRegression(self, coefficients):
        numberOfRows = len(self.x_samplesList)
        # We obtain the predicted data of the desired independent given values
        predictedData = []
        numberOfCoefficients = len(coefficients)-1
        for currentDataPoint in range(0, numberOfRows):
            temporalRow = []
            actualIc = coefficients[0][0]
            for currentIndependentVariable in range(0, numberOfCoefficients):
                actualIc = actualIc + coefficients[currentIndependentVariable+1][0]*(self.x_samplesList[currentDataPoint][0])**(currentIndependentVariable+1)
            temporalRow.append(actualIc)
            predictedData.append(temporalRow)
        # We return the predicted data
        return predictedData
    
    """
    predictMultiplePolynomialRegression(coefficients="We give the Linear Logistic mathematical coefficients that we want to predict with",
                                        orderOfThePolynomial="Assign a whole number that represents the order of degree of the Multiple Polynomial equation you want to make predictions with")
    
    This method returns the predicting values of the independent input values
    that you assign in the local variable of this class: "self.x_samplesList".
    The prediction will be made accordingly to the coefficients and
    configuration specified in the arguments of this method.
    
    CODE EXAMPLE:
        matrix_y = [
                [14.05],
                [10.55],
                [7.55],
                [14.93],
                [9.48],
                [6.59],
                [16.56],
                [13.63],
                [9.23],
                [15.85],
                [11.75],
                [8.78],
                [22.41],
                [18.55],
                [15.93],
                [21.66],
                [17.98],
                [16.44]
                ]
        matrix_x = [
                [75, 15],
                [100, 15],
                [125, 15],
                [75, 17.5],
                [100, 17.5],
                [125, 17.5],
                [75, 20],
                [100, 20],
                [125, 20],
                [75, 22.5],
                [100, 22.5],
                [125, 22.5],
                [75, 25],
                [100, 25],
                [125, 25],
                [75, 27.5],
                [100, 27.5],
                [125, 27.5]
                ]
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        regression = mSL.Regression(matrix_x, matrix_y)
        # "orderOfThePolynomial" = "whole number to represent the desired order of the polynomial model to find"
        # "evtfbmip" stands for "Eliminate Variables To Find Better Model If Possible"
        modelingResults = regression.getMultiplePolynomialRegression(orderOfThePolynomial=4, evtfbmip=True, isClassification=False)
        modelCoefficients = modelingResults[0]
        acurracy = modelingResults[1]
        predictedData = modelingResults[2]
        coefficientDistribution = modelingResults[3]
        allModeledAccuracies = modelingResults[4]
        
        # ------------------------------------- #
        # ----- WE VISUALIZE OUR RESULTS  ----- #
        # ------------------------------------- #
        predictThisValues = [
                [0,1],
                [4,4],
                [6,6],
                [10,10],
                [1,8]
                ]
        regression.set_xSamplesList(predictThisValues)
        predictedValues = regression.predictMultiplePolynomialRegression(coefficients=modelCoefficients, orderOfThePolynomial=4)
        
        
    EXPECTED CODE RESULT:
        predictedValues =
        [[-13.54219748494156],
         [-37.053240090011386],
         [-48.742713747779355],
         [-60.84907570434054],
         [-73.31818590442116]]
    """
    def predictMultiplePolynomialRegression(self, coefficients, orderOfThePolynomial):
        numberOfRows = len(self.x_samplesList)
        # We obtain the predicted data of the desired independent given values
        predictedData = []
        numberOfCoefficients = len(coefficients)
        for row in range(0, numberOfRows):
            temporalRow = []
            actualIc = coefficients[0][0]
            currentOrderOfThePolynomial = 1
            currentVariable = 0
            for currentIndependentVariable in range(0, numberOfCoefficients-1):
                if (currentOrderOfThePolynomial == (orderOfThePolynomial+1)):
                    currentOrderOfThePolynomial = 1
                    currentVariable = currentVariable + 1
                actualIc = actualIc + coefficients[currentIndependentVariable+1][0]*self.x_samplesList[row][currentVariable]**(currentOrderOfThePolynomial)
                currentOrderOfThePolynomial = currentOrderOfThePolynomial + 1
            temporalRow.append(actualIc)
            predictedData.append(temporalRow)
        # We return the predicted data
        return predictedData
    
    """
    predictCustomizedMultipleSecondOrderPolynomialRegression(coefficients="We give the Linear Logistic mathematical coefficients that we want to predict with")
    
    This method returns the predicting values of the independent input values
    that you assign in the local variable of this class: "self.x_samplesList".
    The prediction will be made accordingly to the coefficients and
    configuration specified in the arguments of this method.
    
    CODE EXAMPLE:
        matrix_y = [
                [14.05],
                [10.55],
                [7.55],
                [14.93],
                [9.48],
                [6.59],
                [16.56],
                [13.63],
                [9.23],
                [15.85],
                [11.75],
                [8.78],
                [22.41],
                [18.55],
                [15.93],
                [21.66],
                [17.98],
                [16.44]
                ]
        matrix_x = [
                [75, 15],
                [100, 15],
                [125, 15],
                [75, 17.5],
                [100, 17.5],
                [125, 17.5],
                [75, 20],
                [100, 20],
                [125, 20],
                [75, 22.5],
                [100, 22.5],
                [125, 22.5],
                [75, 25],
                [100, 25],
                [125, 25],
                [75, 27.5],
                [100, 27.5],
                [125, 27.5]
                ]
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        regression = mSL.Regression(matrix_x, matrix_y)
        modelingResults = regression.getCustomizedMultipleSecondOrderPolynomialRegression(evtfbmip = True, isClassification=False)
        modelCoefficients = modelingResults[0]
        acurracy = modelingResults[1]
        predictedData = modelingResults[2]
        coefficientDistribution = modelingResults[3]
        allModeledAccuracies = modelingResults[4]
        
        # ------------------------------------- #
        # ----- WE VISUALIZE OUR RESULTS  ----- #
        # ------------------------------------- #
        predictThisValues = [
                [0,1],
                [4,4],
                [6,6],
                [10,10],
                [1,8]
                ]
        regression.set_xSamplesList(predictThisValues)
        predictedValues = regression.predictCustomizedMultipleSecondOrderPolynomialRegression(coefficients=modelCoefficients)
        
        
    EXPECTED CODE RESULT:
        predictedValues =
        [[39.13047301587551],
         [34.803724444448],
         [32.60300063492485],
         [29.365301587306917],
         [32.832886349211385]]
    """
    def predictCustomizedMultipleSecondOrderPolynomialRegression(self, coefficients):
        numberOfRows = len(self.x_samplesList)
        # We obtain the predicted data of the desired independent given values
        predictedData = []
        for row in range(0, numberOfRows):
            temporalRow = []
            actualIc = coefficients[0][0] + coefficients[1][0]*self.x_samplesList[row][0] + coefficients[2][0]*self.x_samplesList[row][0]**2 + coefficients[3][0]*self.x_samplesList[row][1] + coefficients[4][0]*self.x_samplesList[row][1]**2 + coefficients[5][0]*self.x_samplesList[row][0]*self.x_samplesList[row][1]
            temporalRow.append(actualIc)
            predictedData.append(temporalRow)
        # We return the predicted data
        return predictedData
    
    """
    predictCustomizedMultipleThirdOrderPolynomialRegression(coefficients="We give the Linear Logistic mathematical coefficients that we want to predict with")
    
    This method returns the predicting values of the independent input values
    that you assign in the local variable of this class: "self.x_samplesList".
    The prediction will be made accordingly to the coefficients and
    configuration specified in the arguments of this method.
    
    CODE EXAMPLE:
        matrix_y = [
                [14.05],
                [10.55],
                [7.55],
                [14.93],
                [9.48],
                [6.59],
                [16.56],
                [13.63],
                [9.23],
                [15.85],
                [11.75],
                [8.78],
                [22.41],
                [18.55],
                [15.93],
                [21.66],
                [17.98],
                [16.44]
                ]
        matrix_x = [
                [75, 15],
                [100, 15],
                [125, 15],
                [75, 17.5],
                [100, 17.5],
                [125, 17.5],
                [75, 20],
                [100, 20],
                [125, 20],
                [75, 22.5],
                [100, 22.5],
                [125, 22.5],
                [75, 25],
                [100, 25],
                [125, 25],
                [75, 27.5],
                [100, 27.5],
                [125, 27.5]
                ]
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        regression = mSL.Regression(matrix_x, matrix_y)
        modelingResults = regression.getCustomizedMultipleThirdOrderPolynomialRegression(evtfbmip=True, isClassification=False)
        modelCoefficients = modelingResults[0]
        acurracy = modelingResults[1]
        predictedData = modelingResults[2]
        coefficientDistribution = modelingResults[3]
        allModeledAccuracies = modelingResults[4]
        
        # ------------------------------------- #
        # ----- WE VISUALIZE OUR RESULTS  ----- #
        # ------------------------------------- #
        predictThisValues = [
                [0,1],
                [4,4],
                [6,6],
                [10,10],
                [1,8]
                ]
        regression.set_xSamplesList(predictThisValues)
        predictedValues = regression.predictCustomizedMultipleThirdOrderPolynomialRegression(coefficients=modelCoefficients)
        
        
    EXPECTED CODE RESULT:
        predictedValues =
        [[105.28333074423442],
         [72.81181980293967],
         [56.899154811293464],
         [36.45941710222553],
         [46.042387049575304]]
    """
    def predictCustomizedMultipleThirdOrderPolynomialRegression(self, coefficients):
        numberOfRows = len(self.x_samplesList)
        # We obtain the predicted data of the desired independent given values
        predictedData = []
        x1 = 0
        x2 = 1
        for row in range(0, numberOfRows):
            temporalRow = []
            actualIc = coefficients[0][0] + coefficients[1][0]*self.x_samplesList[row][x1] + coefficients[2][0]*self.x_samplesList[row][x1]**2 + coefficients[3][0]*self.x_samplesList[row][x1]**3 + coefficients[4][0]*self.x_samplesList[row][x2] + coefficients[5][0]*self.x_samplesList[row][x2]**2 + coefficients[6][0]*self.x_samplesList[row][x2]**3 + coefficients[7][0]*self.x_samplesList[row][x1]*self.x_samplesList[row][x2] + coefficients[8][0]*(self.x_samplesList[row][x1]**2)*self.x_samplesList[row][x2] + coefficients[9][0]*self.x_samplesList[row][x1]*(self.x_samplesList[row][x2]**2)
            temporalRow.append(actualIc)
            predictedData.append(temporalRow)
        # We return the predicted data
        return predictedData
    
    
"""
Classification("x independent variable datapoints to model", "y dependent variable datapoints to model")

The Classification Library gives several methods to be able to get the best
fitting classification model to predict a determined classification problem.
"""    
class Classification:
    def __init__(self, x_samplesList, y_samplesList):
        self.y_samplesList = y_samplesList
        self.x_samplesList = x_samplesList
    
    def set_xSamplesList(self, x_samplesList):
        self.x_samplesList = x_samplesList
        
    def set_ySamplesList(self, y_samplesList):
        self.y_samplesList = y_samplesList
        
    """
    getSupportVectorMachine(evtfbmip="True to indicate to Eliminate Variables To Find Better Model If Possible. False if the contrary is desired")
    
    This method returns the best fitting Linear Support Vector Machine model to
    be able to predict a classification problem of any number of independent
    variables (x).
    
    CODE EXAMPLE:
        matrix_x = [
             [0, 0],
             [2, 2],
             [4, 3],
             [2, 4],
             [3, 4],
             [4, 4],
             [5, 3],
             [3, 5],
             [4, 6]
             ]
        
        matrix_y = [
             [1],
             [1],
             [1],
             [1],
             [-1],
             [-1],
             [-1],
             [-1],
             [-1]
             ]
        
        classification = mSL.Classification(matrix_x, matrix_y)
        # evtfbmip stands for "Eliminate Variables To Find Better Model If Possible"
        modelingResults = classification.getSupportVectorMachine(evtfbmip = True)
        modelCoefficients = modelingResults[0]
        acurracy = modelingResults[1]
        predictedData = modelingResults[2]
        coefficientDistribution = modelingResults[3]
        allModeledAccuracies = modelingResults[4]
        
    EXPECTED CODE RESULT:
        modelCoefficients =
        [[1.5736095873424212], [-0.26050769870994606], [-0.25468164794007475]]
        
        acurracy =
        88.88888888888889
        
        predictedData = [
                     [1],
                     [1],
                     [-1],
                     [1],
                     [-1],
                     [-1],
                     [-1],
                     [-1],
                     [-1]
                     ]
        
        coefficientDistribution =
        'Coefficients distribution is as follows: b1*x1 + b2*x2 + ... + bn*xn >= -bo (As a note, remember that true equation representation is:   w.x>=c)'
        
        allModeledAccuracies["independent variable distribution used to get a model"]["model accuracy", "model coefficients obtained but with original distribution", "matrix x data"] =
        # NOTE: since this variable contains large amounts of information, it
        #       will not be displayed but only described on how to use it.
    """
    def getSupportVectorMachine(self, evtfbmip=True):
        getOptimizedRegression = evtfbmip
        numberOfRows = len(self.y_samplesList)
        matrix_x = self.x_samplesList
        matrix_y = self.y_samplesList
        for row in range(0, numberOfRows):
            if ((self.y_samplesList[row][0]!=1) and (self.y_samplesList[row][0]!=-1)):
                raise Exception('ERROR: One of the dependent (y) data points does not have exactly a 1 or a -1 as value. Note that in this API, the Support Vector Machine method needs to process your data to have either +1 or -1 as values.')
        # We apply a Multiple Linear Regression to get the coefficient values
        # for our Linear Support Vector Machine Model
        from . import MortrackML_Library as mSL
        import math
        regression = mSL.Regression(matrix_x, matrix_y)
        modelingResults = regression.getMultipleLinearRegression(evtfbmip = getOptimizedRegression)
        svcCoefficients = modelingResults[0]
        svcPredictedData = modelingResults[2]
        
        # ---------------------------------- #
        # ----- b0 Coefficient Tunning ----- #
        # ---------------------------------- #
        # Through the best fitting Multiple Linear Regression, we make a
        # new search to try to find a better fitting b0 coefficient value
        # that best fits the conditional of the equation that we actually
        # want to solve (w.x>=-b0)
        import numpy as np
        rangeOfPredictedData = max(svcPredictedData)[0] - min(svcPredictedData)[0]
        # linspace(start, stop, num=50)
        bStepValues = np.linspace(svcCoefficients[0][0]-rangeOfPredictedData, svcCoefficients[0][0]+rangeOfPredictedData, num=100)
        numberOfCoefficients = len(svcCoefficients)
        best_b_value = 0
        bestPredictedData = 0
        bestPredictionAccuracy = 0
        # We first get the b value that first pops and that has the highest
        # accuracy
        for currentStepValue in range(0, len(bStepValues)):
            current_b_value = bStepValues[currentStepValue]
            # We get the predicted data with the trained Kernel Support Vector
            # Classification (K-SVC) model
            predictedData = []
            for row in range(0, numberOfRows):
                temporalRow = []
                wx = 0
                for column in range(0, numberOfCoefficients-1):
                    wx = wx + (matrix_x[row][column])*svcCoefficients[column+1][0]
                c = -current_b_value # c=ln(y=0)-b0
                if (wx >= c):
                    temporalRow.append(1) # Its a positive sample
                else:
                    temporalRow.append(-1) # Its a negative sample
                predictedData.append(temporalRow)
            
            predictionAcurracy = 0
            n2 = 0
            n1 = 0
            for row in range(0, numberOfRows):
                n2 = self.y_samplesList[row][0]
                n1 = predictedData[row][0]
                if (n1 == n2):
                    predictionAcurracy = predictionAcurracy + 1
            predictionAcurracy = predictionAcurracy/numberOfRows*100
            if (predictionAcurracy > bestPredictionAccuracy):
                best_b_value = current_b_value
                bestPredictedData = predictedData
                bestPredictionAccuracy = predictionAcurracy
        
        # Now that we now what value of b0 gives the best accuracy, we look
        # forward to find the range of the b0 values that gives such best
        # accuracy
        best_b_value_1 = best_b_value
        best_b_value_2 = 0
        isBest_b_value = False
        for currentStepValue in range(0, len(bStepValues)):
            current_b_value = bStepValues[currentStepValue]
            if (current_b_value == best_b_value_1):
                isBest_b_value = True
            if (isBest_b_value == True):
                # We get the predicted data with the trained Kernel Support Vector
                # Classification (K-SVC) model
                predictedData = []
                for row in range(0, numberOfRows):
                    temporalRow = []
                    wx = 0
                    for column in range(0, numberOfCoefficients-1):
                        wx = wx + (matrix_x[row][column])*svcCoefficients[column+1][0]
                    c = -current_b_value # c=ln(y=0)-b0
                    if (wx >= c):
                        temporalRow.append(1) # Its a positive sample
                    else:
                        temporalRow.append(-1) # Its a negative sample
                    predictedData.append(temporalRow)
                
                predictionAcurracy = 0
                n2 = 0
                n1 = 0
                for row in range(0, numberOfRows):
                    n2 = self.y_samplesList[row][0]
                    n1 = predictedData[row][0]
                    if (n1 == n2):
                        predictionAcurracy = predictionAcurracy + 1
                predictionAcurracy = predictionAcurracy/numberOfRows*100
                if (predictionAcurracy == bestPredictionAccuracy):
                    best_b_value_2 = current_b_value
        # We find best fitting b0 coefficient value through exponential
        # method
        b0_sign = 1
        if ((best_b_value_1+best_b_value_2)<0):
            b0_sign = -1
        best_b_value = (math.log(abs(best_b_value_1)) + math.log(abs(best_b_value_2)))/2
        best_b_value = b0_sign*math.exp(best_b_value)
        # We get the predicted data with the trained Kernel Support Vector
        # Classification (K-SVC) model
        predictedData = []
        for row in range(0, numberOfRows):
            temporalRow = []
            wx = 0
            for column in range(0, numberOfCoefficients-1):
                wx = wx + (matrix_x[row][column])*svcCoefficients[column+1][0]
            c = -current_b_value # c=ln(y=0)-b0
            if (wx >= c):
                temporalRow.append(1) # Its a positive sample
            else:
                temporalRow.append(-1) # Its a negative sample
            predictedData.append(temporalRow)
        predictionAcurracy = 0
        n2 = 0
        n1 = 0
        for row in range(0, numberOfRows):
            n2 = self.y_samplesList[row][0]
            n1 = predictedData[row][0]
            if (n1 == n2):
                predictionAcurracy = predictionAcurracy + 1
        predictionAcurracy = predictionAcurracy/numberOfRows*100
        
        # We verify if exponential method was the best choice to pick best
        # fitting b0 coefficient. If this isnt true, we then try again but
        # with the mean value of the b0 coefficient range that we obtained
        #earlier
        if ((best_b_value<min([best_b_value_1, best_b_value_2])) or (best_b_value>max([best_b_value_1, best_b_value_2])) or (predictionAcurracy<bestPredictionAccuracy)):
            best_b_value = (best_b_value_1+best_b_value_2)/2
            # We get the predicted data with the trained Kernel Support Vector
            # Classification (K-SVC) model
            predictedData = []
            for row in range(0, numberOfRows):
                temporalRow = []
                wx = 0
                for column in range(0, numberOfCoefficients-1):
                    wx = wx + (matrix_x[row][column])*svcCoefficients[column+1][0]
                c = -current_b_value # c=ln(y=0)-b0
                if (wx >= c):
                    temporalRow.append(1) # Its a positive sample
                else:
                    temporalRow.append(-1) # Its a negative sample
                predictedData.append(temporalRow)
            predictionAcurracy = 0
            n2 = 0
            n1 = 0
            for row in range(0, numberOfRows):
                n2 = self.y_samplesList[row][0]
                n1 = predictedData[row][0]
                if (n1 == n2):
                    predictionAcurracy = predictionAcurracy + 1
            predictionAcurracy = predictionAcurracy/numberOfRows*100
            # If neither the exponential nor the mean methods work to get the
            # best fitting b0 coefficient value, we then just pick the initial
            # best fitting b0 value that we identified in this algorithm
            if (predictionAcurracy < bestPredictionAccuracy):
                best_b_value = best_b_value_1
        
        # We save the new-found b0 coefficient value that best fits our
        # current dataset
        svcCoefficients[0][0] = best_b_value
        
        # ----------------------------------------- #
        # ----- We save best modeling results ----- #
        # ----------------------------------------- #
        # We save the current the modeling results
        bestModelingResults = []
        bestModelingResults.append(svcCoefficients)
        bestModelingResults.append(bestPredictionAccuracy)
        bestModelingResults.append(bestPredictedData)
        bestModelingResults.append("Coefficients distribution is as follows: b1*x1 + b2*x2 + ... + bn*xn >= -bo (As a note, remember that true equation representation is:   w.x>=c)")
        allAccuracies = []
        temporalRow = []
        temporalRow.append(bestModelingResults[1])
        temporalRow.append(bestModelingResults[0])
        temporalRow.append(self.x_samplesList)
        allAccuracies.append(temporalRow)
        bestModelingResults.append(allAccuracies)
        return bestModelingResults
        
        
    """
    getKernelSupportVectorMachine(kernel="you specify here the type of kernel that you want to model with. literally write, in strings, gaussian for a gaussian kernel; polynomial for a polynomial kernel; and linear for a linear kernel",
                                  isPolynomialSVC="True if you want to apply a polynomial SVC. False if otherwise is desired",
                                  orderOfPolynomialSVC="If you apply a polynomial SVC through the argument isPolynomialSVC, you then give a whole number here to indicate the order of degree that you desire in such Polynomial SVC",
                                  orderOfPolynomialKernel="if you selected polynomial kernel in the kernel argument, you then here give a whole number to indicate the order of degree that you desire in such Polynomial Kernel",
                                  evtfbmip="True to indicate to Eliminate Variables To Find Better Model If Possible. False if the contrary is desired")
    
    
    This method returns the best fitting Kernel Support Vector Machine
    model to be able to predict a classification problem of any number of
    independent variables (x).
    * If "gaussian" kernel is applied. This method will find the best
      fitting model of such gaussian kernel through a gaussian regression.
    * If "polynomimal" kernel is applied. This method will find the best
      fitting model of such polynomial kernel through a Multiple Polynomial
      Regression. You can specify the order of degree that you desire for your
      Multiple Polynomial Kernel through the argument of this method named as
      "orderOfPolynomialKernel".
    * If "linear" kernel is applied. This method will find the best fitting
      model of such polynomial kernel through a Multiple Linear Regression.
    * You can also get a modified SVC by getting a non-linear intersection
      plane to split your dataset into 2 specified categories. If you apply
      this modified SVC, through "isPolynomialSVC" argument of this method,
      you will be able to get a polynomial intersecting plane for your dataset
      whos degree order can be modified through the argument of this method
      named as "orderOfPolynomialSVC".
    
    CODE EXAMPLE:
        matrix_x = [
             [0, 0],
             [2, 2],
             [4, 3],
             [2, 4],
             [3, 4],
             [4, 4],
             [5, 3],
             [3, 5],
             [4, 6]
             ]
        
        matrix_y = [
             [1],
             [1],
             [1],
             [1],
             [-1],
             [-1],
             [-1],
             [-1],
             [-1]
             ]
        
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        classification = mSL.Classification(matrix_x, matrix_y)
        # evtfbmip stands for "Eliminate Variables To Find Better Model If Possible"
        modelingResults = classification.getKernelSupportVectorMachine(kernel='gaussian', isPolynomialSVC=True, orderOfPolynomialSVC=2, orderOfPolynomialKernel=3, evtfbmip=True)
        modelCoefficients = modelingResults[0]
        acurracy = modelingResults[1]
        predictedData = modelingResults[2]
        coefficientDistribution = modelingResults[3]
        allModeledAccuracies = modelingResults[4]
        
    EXPECTED CODE RESULT:
        modelCoefficients =
        [
         [
          [-0.4067247938936074],
          [-2.638275880744686],
          [0.6025816805607462],
          [1.5978782207152165],
          [0.0018850313260649898]
         ],
         [
          [17.733125277353782],
          [-0.41918858713133034],
          [-0.07845753695120994],
          [-7.126885817943787],
          [0.7414460867570138],
          [13.371724079069963],
          [-16.435714646771032]
         ]
        ]
        
        acurracy =
        100.0
        
        predictedData = [
                     [1],
                     [1],
                     [1],
                     [1],
                     [-1],
                     [-1],
                     [-1],
                     [-1],
                     [-1]
                     ]
        
        coefficientDistribution =
        [
         'Coefficients distribution for the Gaussian Kernel is as follows:    kernel = exp(-(bo + b1*x1 + b2*x1^2 + b3*x2 + b4*x2^2 + ... + b_(n-1)*xn + bn*xn^2))',
         [
          'Coefficients distribution is as follows: b1*x1 + b2*x2 + ... + b_(n-1)*xn + bn*Kernel >= -b_0  --> for linear SVC (As a note, remember that true equation representation is:   w.x>=c and that x here represents each one of the coordinates of your independent samples (x))',
          'Coefficients distribution is as follows: b1*x1 + ... + b_(n-5)*x_m^m + b_(n-4)*x_(m-1) + ... + b_(n-3)*x_m^m + ... + b_(n-2)*x_m + ... + b_(n-1)*x_m^m + bn*Kernel >= -b_0  --> for polynomial SVC (m stands for the order degree selected for the polynomial SVC and n stands for the number of coefficients used in the polynomial SVC)'
         ]
        ]
        
        allModeledAccuracies["independent variable distribution used to get a model"]["model accuracy", "model coefficients obtained but with original distribution", "matrix x data"] =
        # NOTE: since this variable contains large amounts of information, it
        #       will not be displayed but only described on how to use it.
    """
    def getKernelSupportVectorMachine(self, kernel, isPolynomialSVC=True, orderOfPolynomialSVC=3, orderOfPolynomialKernel=3, evtfbmip=True):
        if ((kernel!='linear') and (kernel!='polynomial') and (kernel!='gaussian')):
            raise Exception('ERROR: The selected Kernel does not exist or has not been programmed in this method yet.')
        from . import MortrackML_Library as mSL
        import math
        getOptimizedRegression = evtfbmip
        numberOfRows = len(self.y_samplesList)
        
        # --------------------------- #
        # ----- Kernel Tranning ----- #
        # --------------------------- #
        if (kernel=='gaussian'):
            # We obtain the independent coefficients of the best fitting model
            # obtained through the Gaussian function (kernel) that we will use to distort
            # the current dimentional spaces that we were originally given by the
            # user
            regression = mSL.Regression(self.x_samplesList, self.y_samplesList)
            modelingResults = regression.getGaussianRegression()
            kernelCoefficients = modelingResults[0]
            
            # We obtain the coordinates of only the new dimentional space created
            # by the obtained kernel
            kernelData = []
            numberOfCoefficients = len(kernelCoefficients)
            gaussAproximationOrder = 2
            for row in range(0, numberOfRows):
                temporalRow = []
                actualIc = kernelCoefficients[0][0]
                currentOrderOfThePolynomial = 1
                currentVariable = 0
                for currentIndependentVariable in range(0, numberOfCoefficients-1):
                    if (currentOrderOfThePolynomial == (gaussAproximationOrder+1)):
                        currentOrderOfThePolynomial = 1
                        currentVariable = currentVariable + 1
                    actualIc = actualIc + kernelCoefficients[currentIndependentVariable+1][0]*self.x_samplesList[row][currentVariable]**(currentOrderOfThePolynomial)
                    currentOrderOfThePolynomial = currentOrderOfThePolynomial + 1
                temporalRow.append(math.exp(-actualIc))
                kernelData.append(temporalRow)
            
        if (kernel=='polynomial'):
            # We obtain the independent coefficients of the best fitting model
            # obtained through the Multiple Polynomial Regression function
            # (kernel) that we will use to distort the current dimentional
            # spaces that we were originally given by the user
            regression = mSL.Regression(self.x_samplesList, self.y_samplesList)
            modelingResults = regression.getMultiplePolynomialRegression(orderOfThePolynomial=orderOfPolynomialKernel, evtfbmip=getOptimizedRegression)
            kernelCoefficients = modelingResults[0]
            
            # We obtain the predicted data through the current obtained
            # coefficients
            kernelData = []
            numberOfCoefficients = len(kernelCoefficients)
            for row in range(0, numberOfRows):
                temporalRow = []
                actualIc = kernelCoefficients[0][0]
                currentOrderOfThePolynomial = 1
                currentVariable = 0
                for currentIndependentVariable in range(0, numberOfCoefficients-1):
                    if (currentOrderOfThePolynomial == (orderOfPolynomialKernel+1)):
                        currentOrderOfThePolynomial = 1
                        currentVariable = currentVariable + 1
                    actualIc = actualIc + kernelCoefficients[currentIndependentVariable+1][0]*self.x_samplesList[row][currentVariable]**(currentOrderOfThePolynomial)
                    currentOrderOfThePolynomial = currentOrderOfThePolynomial + 1
                temporalRow.append(actualIc)
                kernelData.append(temporalRow)
            
        if (kernel=='linear'):
            # We obtain the independent coefficients of the best fitting model
            # obtained through the Multiple Linear Regression function
            # (kernel) that we will use to distort the current dimentional
            # spaces that we were originally given by the user
            regression = mSL.Regression(self.x_samplesList, self.y_samplesList)
            modelingResults = regression.getMultipleLinearRegression(evtfbmip=getOptimizedRegression)
            kernelCoefficients = modelingResults[0]
            
            # We obtain the predicted data through the current obtained
            # coefficients
            kernelData = []
            numberOfIndependentVariables = len(self.x_samplesList[0])
            for row in range(0, numberOfRows):
                temporalRow = []
                actualIc = kernelCoefficients[0][0]
                for currentIndependentVariable in range(0, numberOfIndependentVariables):
                    actualIc = actualIc + kernelCoefficients[currentIndependentVariable+1][0]*self.x_samplesList[row][currentIndependentVariable]
                temporalRow.append(actualIc)
                kernelData.append(temporalRow)
        
        # We create the new matrix of the independent variables (x) but with
        # the new dimentional space distortion made by the Kernel created
        newMatrix_x = []
        for row in range(0, numberOfRows):
            temporalRow = []
            for column in range(0, len(self.x_samplesList[0])):
                temporalRow.append(self.x_samplesList[row][column])
            temporalRow.append(kernelData[row][0])
            newMatrix_x.append(temporalRow)
        
        # ----------------------------------------------- #
        # ----- Support Vector Classifier Trainning ----- #
        # ----------------------------------------------- #
        # We apply a Multiple Linear Regression to get the coefficient values
        # for our Linear Support Vector Machine Model (Its Linear: remember
        # that we applied a Kernel to distort the original dimentional space so
        # that we could gain a linearly modeable dataset)
        regression = mSL.Regression(newMatrix_x, self.y_samplesList)
        if (isPolynomialSVC==True):
            modelingResults = regression.getMultiplePolynomialRegression(orderOfThePolynomial=orderOfPolynomialSVC, evtfbmip=getOptimizedRegression)
        else:
            modelingResults = regression.getMultipleLinearRegression(evtfbmip = getOptimizedRegression)
        svcCoefficients = modelingResults[0]
        svcPredictedData = modelingResults[2]
        
        # ---------------------------------- #
        # ----- b0 Coefficient Tunning ----- #
        # ---------------------------------- #
        # Through the best fitting Multiple Linear Regression, we make a
        # new search to try to find a better fitting b0 coefficient value
        # that best fits the conditional of the equation that we actually
        # want to solve (w.x>=-b0)
        import numpy as np
        rangeOfPredictedData = max(svcPredictedData)[0] - min(svcPredictedData)[0]
        # linspace(start, stop, num=50)
        bStepValues = np.linspace(svcCoefficients[0][0]-rangeOfPredictedData, svcCoefficients[0][0]+rangeOfPredictedData, num=100)
        numberOfCoefficients = len(svcCoefficients)
        best_b_value = 0
        bestPredictedData = 0
        bestPredictionAccuracy = 0
        # We first get the b value that first pops and that has the highest
        # accuracy
        for currentStepValue in range(0, len(bStepValues)):
            current_b_value = bStepValues[currentStepValue]
            # We get the predicted data with the trained Kernel Support Vector
            # Classification (K-SVC) model
            predictedData = []
            for row in range(0, numberOfRows):
                temporalRow = []
                wx = 0
                if (isPolynomialSVC==True):
                    currentOrderOfThePolynomial = 1
                    currentVariable = 0
                    for currentIndependentVariable in range(0, numberOfCoefficients-1):
                        if (currentOrderOfThePolynomial == (orderOfPolynomialSVC+1)):
                            currentOrderOfThePolynomial = 1
                            currentVariable = currentVariable + 1
                        wx = wx + svcCoefficients[currentIndependentVariable+1][0]*newMatrix_x[row][currentVariable]**(currentOrderOfThePolynomial)
                        currentOrderOfThePolynomial = currentOrderOfThePolynomial + 1
                else:
                    for column in range(0, numberOfCoefficients-1):
                        wx = wx + (newMatrix_x[row][column])*svcCoefficients[column+1][0]
                c = -current_b_value # c=ln(y=0)-b0
                if (wx >= c):
                    temporalRow.append(1) # Its a positive sample
                else:
                    temporalRow.append(-1) # Its a negative sample
                predictedData.append(temporalRow)
            
            predictionAcurracy = 0
            n2 = 0
            n1 = 0
            for row in range(0, numberOfRows):
                n2 = self.y_samplesList[row][0]
                n1 = predictedData[row][0]
                if (n1 == n2):
                    predictionAcurracy = predictionAcurracy + 1
            predictionAcurracy = predictionAcurracy/numberOfRows*100
            if (predictionAcurracy > bestPredictionAccuracy):
                best_b_value = current_b_value
                bestPredictedData = predictedData
                bestPredictionAccuracy = predictionAcurracy
        
        # Now that we now what value of b0 gives the best accuracy, we look
        # forward to find the range of the b0 values that gives such best
        # accuracy
        best_b_value_1 = best_b_value
        best_b_value_2 = 0
        isBest_b_value = False
        for currentStepValue in range(0, len(bStepValues)):
            current_b_value = bStepValues[currentStepValue]
            if (current_b_value == best_b_value_1):
                isBest_b_value = True
            if (isBest_b_value == True):
                # We get the predicted data with the trained Kernel Support Vector
                # Classification (K-SVC) model
                predictedData = []
                for row in range(0, numberOfRows):
                    temporalRow = []
                    wx = 0
                    if (isPolynomialSVC==True):
                        currentOrderOfThePolynomial = 1
                        currentVariable = 0
                        for currentIndependentVariable in range(0, numberOfCoefficients-1):
                            if (currentOrderOfThePolynomial == (orderOfPolynomialSVC+1)):
                                currentOrderOfThePolynomial = 1
                                currentVariable = currentVariable + 1
                            wx = wx + svcCoefficients[currentIndependentVariable+1][0]*newMatrix_x[row][currentVariable]**(currentOrderOfThePolynomial)
                            currentOrderOfThePolynomial = currentOrderOfThePolynomial + 1
                    else:
                        for column in range(0, numberOfCoefficients-1):
                            wx = wx + (newMatrix_x[row][column])*svcCoefficients[column+1][0]
                    c = -current_b_value # c=ln(y=0)-b0
                    if (wx >= c):
                        temporalRow.append(1) # Its a positive sample
                    else:
                        temporalRow.append(-1) # Its a negative sample
                    predictedData.append(temporalRow)
                
                predictionAcurracy = 0
                n2 = 0
                n1 = 0
                for row in range(0, numberOfRows):
                    n2 = self.y_samplesList[row][0]
                    n1 = predictedData[row][0]
                    if (n1 == n2):
                        predictionAcurracy = predictionAcurracy + 1
                predictionAcurracy = predictionAcurracy/numberOfRows*100
                if (predictionAcurracy == bestPredictionAccuracy):
                    best_b_value_2 = current_b_value
        # We find best fitting b0 coefficient value through exponential
        # method
        b0_sign = 1
        if ((best_b_value_1+best_b_value_2)<0):
            b0_sign = -1
        best_b_value = (math.log(abs(best_b_value_1)) + math.log(abs(best_b_value_2)))/2
        best_b_value = b0_sign*math.exp(best_b_value)
        # We get the predicted data with the trained Kernel Support Vector
        # Classification (K-SVC) model
        predictedData = []
        for row in range(0, numberOfRows):
            temporalRow = []
            wx = 0
            if (isPolynomialSVC==True):
                currentOrderOfThePolynomial = 1
                currentVariable = 0
                for currentIndependentVariable in range(0, numberOfCoefficients-1):
                    if (currentOrderOfThePolynomial == (orderOfPolynomialSVC+1)):
                        currentOrderOfThePolynomial = 1
                        currentVariable = currentVariable + 1
                    wx = wx + svcCoefficients[currentIndependentVariable+1][0]*newMatrix_x[row][currentVariable]**(currentOrderOfThePolynomial)
                    currentOrderOfThePolynomial = currentOrderOfThePolynomial + 1
            else:
                for column in range(0, numberOfCoefficients-1):
                    wx = wx + (newMatrix_x[row][column])*svcCoefficients[column+1][0]
            c = -best_b_value # c=ln(y=0)-b0
            if (wx >= c):
                temporalRow.append(1) # Its a positive sample
            else:
                temporalRow.append(-1) # Its a negative sample
            predictedData.append(temporalRow)
        predictionAcurracy = 0
        n2 = 0
        n1 = 0
        for row in range(0, numberOfRows):
            n2 = self.y_samplesList[row][0]
            n1 = predictedData[row][0]
            if (n1 == n2):
                predictionAcurracy = predictionAcurracy + 1
        predictionAcurracy = predictionAcurracy/numberOfRows*100
        
        # We verify if exponential method was the best choice to pick best
        # fitting b0 coefficient. If this isnt true, we then try again but
        # with the mean value of the b0 coefficient range that we obtained
        #earlier
        if ((best_b_value<min([best_b_value_1, best_b_value_2])) or (best_b_value>max([best_b_value_1, best_b_value_2])) or (predictionAcurracy<bestPredictionAccuracy)):
            best_b_value = (best_b_value_1+best_b_value_2)/2
            # We get the predicted data with the trained Kernel Support Vector
            # Classification (K-SVC) model
            predictedData = []
            for row in range(0, numberOfRows):
                temporalRow = []
                wx = 0
                if (isPolynomialSVC==True):
                    currentOrderOfThePolynomial = 1
                    currentVariable = 0
                    for currentIndependentVariable in range(0, numberOfCoefficients-1):    
                        if (currentOrderOfThePolynomial == (orderOfPolynomialSVC+1)):
                            currentOrderOfThePolynomial = 1
                            currentVariable = currentVariable + 1
                        wx = wx + svcCoefficients[currentIndependentVariable+1][0]*newMatrix_x[row][currentVariable]**(currentOrderOfThePolynomial)
                        currentOrderOfThePolynomial = currentOrderOfThePolynomial + 1
                else:
                    for column in range(0, numberOfCoefficients-1):
                        wx = wx + (newMatrix_x[row][column])*svcCoefficients[column+1][0]
                c = -best_b_value # c=ln(y=0)-b0
                if (wx >= c):
                    temporalRow.append(1) # Its a positive sample
                else:
                    temporalRow.append(-1) # Its a negative sample
                predictedData.append(temporalRow)
            predictionAcurracy = 0
            n2 = 0
            n1 = 0
            for row in range(0, numberOfRows):
                n2 = self.y_samplesList[row][0]
                n1 = predictedData[row][0]
                if (n1 == n2):
                    predictionAcurracy = predictionAcurracy + 1
            predictionAcurracy = predictionAcurracy/numberOfRows*100
            # If neither the exponential nor the mean methods work to get the
            # best fitting b0 coefficient value, we then just pick the initial
            # best fitting b0 value that we identified in this algorithm
            if (predictionAcurracy < bestPredictionAccuracy):
                best_b_value = best_b_value_1
        
        # We save the new-found b0 coefficient value that best fits our
        # current dataset
        svcCoefficients[0][0] = best_b_value
        
        # ----------------------------------------- #
        # ----- We save best modeling results ----- #
        # ----------------------------------------- #
        # We save the current the modeling results
        bestModelingResults = []
        bestModelingResults.append([kernelCoefficients, svcCoefficients])
        bestModelingResults.append(bestPredictionAccuracy)
        bestModelingResults.append(bestPredictedData)
        temporalRow = []
        if (kernel=='gaussian'):
            temporalRow.append("Coefficients distribution for the Gaussian Kernel is as follows:    kernel = exp(-(bo + b1*x1 + b2*x1^2 + b3*x2 + b4*x2^2 + ... + b_(n-1)*xn + bn*xn^2))")
        if (kernel=='polynomial'):
            temporalRow.append("Coefficients distribution for the Multiple Polynomial Kernel is as follows:    kernel = bo + b1*x1 + b2*x1^2 + ... + bn*x1^n + b3*x2 + b4*x2^2 + ... + bn*x2^n + b5*x3 + b6*x3^2 + ... + bn*xn^n")
        if (kernel=='linear'):
            temporalRow.append("Coefficients distribution for the Multiple Linear Kernel is as follows:    kernel = bo + b1*x1 + b2*x2 + b3*x3 + ... + bn*xn")
        temporalRow2 = []
        temporalRow2.append("Coefficients distribution is as follows: b1*x1 + b2*x2 + ... + b_(n-1)*xn + bn*Kernel >= -b_0  --> for linear SVC (As a note, remember that true equation representation is:   w.x>=c and that x here represents each one of the coordinates of your independent samples (x))")
        temporalRow2.append("Coefficients distribution is as follows: b1*x1 + ... + b_(n-5)*x_m^m + b_(n-4)*x_(m-1) + ... + b_(n-3)*x_m^m + ... + b_(n-2)*x_m + ... + b_(n-1)*x_m^m + bn*Kernel >= -b_0  --> for polynomial SVC (m stands for the order degree selected for the polynomial SVC and n stands for the number of coefficients used in the polynomial SVC)")
        temporalRow.append(temporalRow2)
        bestModelingResults.append(temporalRow)
        allAccuracies = []
        temporalRow = []
        temporalRow.append(bestModelingResults[1])
        temporalRow.append(bestModelingResults[0])
        temporalRow.append(self.x_samplesList)
        allAccuracies.append(temporalRow)
        bestModelingResults.append(allAccuracies)
        return bestModelingResults
    
    """
    predictSupportVectorMachine(coefficients="We give the SVC mathematical coefficients that we want to predict with")
    
    This method returns the predicting values of the independent input values
    that you assign in the local variable of this class: "self.x_samplesList".
    The prediction will be made accordingly to the coefficients and
    configuration specified in the arguments of this method.
    
    CODE EXAMPLE:
        matrix_x = [
             [0, 0],
             [2, 2],
             [4, 3],
             [2, 4],
             [3, 4],
             [4, 4],
             [5, 3],
             [3, 5],
             [4, 6]
             ]
        
        matrix_y = [
             [1],
             [1],
             [1],
             [1],
             [-1],
             [-1],
             [-1],
             [-1],
             [-1]
             ]
        
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        classification = mSL.Classification(matrix_x, matrix_y)
        # evtfbmip stands for "Eliminate Variables To Find Better Model If Possible"
        modelingResults = classification.getSupportVectorMachine(evtfbmip = True)
        modelCoefficients = modelingResults[0]
        acurracy = modelingResults[1]
        predictedData = modelingResults[2]
        coefficientDistribution = modelingResults[3]
        allModeledAccuracies = modelingResults[4]
        
        # --------------------------------------------------------------------------- #
        # ----- WE VISUALIZE OUR RESULTS: CBES WHEN VOLTAGE APPLIED IS POSITIVE ----- #
        # --------------------------------------------------------------------------- #
        # Visualising the Training set results
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import ListedColormap
        plt.figure()
            
        
        # We plot the Background
        x1_samples = []
        x2_samples = []
        for row in range(0, len(matrix_x)):
            x1_samples.append(matrix_x[row][0])
            x2_samples.append(matrix_x[row][1])
        # linspace(start, stop, num=50)
        x1_distance = min(x1_samples) - max(x1_samples)
        x2_distance = min(x2_samples) - max(x2_samples)
        x1_background = np.linspace(min(x1_samples)+x1_distance*0.1, max(x1_samples)-x1_distance*0.1, num=100)
        x2_background = np.linspace(min(x2_samples)+x2_distance*0.1, max(x2_samples)-x2_distance*0.1, num=100)
        predictThisValues = []
        for row in range(0, len(x1_background)):
            for row2 in range(0, len(x2_background)):
                temporalRow = []
                temporalRow.append(x1_background[row])
                temporalRow.append(x2_background[row2])
                predictThisValues.append(temporalRow)
        classification.set_xSamplesList(predictThisValues)
        predictedValuesForBg = classification.predictSupportVectorMachine(coefficients=modelCoefficients)
        positives_x = []
        positives_y = []
        negatives_x = []
        negatives_y = []
        for row in range(0, len(predictedValuesForBg)):
            temporalRow = []
            if (predictedValuesForBg[row][0] == 1):
                temporalRow = []
                temporalRow.append(predictThisValues[row][1])
                positives_y.append(temporalRow)
                temporalRow = []
                temporalRow.append(predictThisValues[row][0])
                positives_x.append(temporalRow)
            else:
                temporalRow = []
                temporalRow.append(predictThisValues[row][1])
                negatives_y.append(temporalRow)
                temporalRow = []
                temporalRow.append(predictThisValues[row][0])
                negatives_x.append(temporalRow)
        plt.scatter(positives_x, positives_y, c='green', s=10, label='predicted positives (1)', alpha = 0.1)
        plt.scatter(negatives_x, negatives_y, c='red', s=10, label='predicted negatives (-1)', alpha = 0.1)
        
        
        # We plot the predicted values of our currently trained model
        positives_x = []
        positives_y = []
        negatives_x = []
        negatives_y = []
        for row in range(0, len(matrix_y)):
            temporalRow = []
            if (matrix_y[row][0] == 1):
                temporalRow = []
                temporalRow.append(matrix_x[row][1])
                positives_y.append(temporalRow)
                temporalRow = []
                temporalRow.append(matrix_x[row][0])
                positives_x.append(temporalRow)
            else:
                temporalRow = []
                temporalRow.append(matrix_x[row][1])
                negatives_y.append(temporalRow)
                temporalRow = []
                temporalRow.append(matrix_x[row][0])
                negatives_x.append(temporalRow)
        plt.scatter(positives_x, positives_y, c='green', s=50, label='real positives (1)')
        plt.scatter(negatives_x, negatives_y, c='red', s=50, label='real negatives (-1)')
        # Finnally, we define the desired title, the labels and the legend for the data
        # points
        plt.title('Real Results')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.grid()
        # We show the graph with all the specifications we just declared.
        plt.show()
        
        
    EXPECTED CODE RESULT:
        "A graph will pop and will show the predicted region of the obtained
         model and the scattered points of the true/real results to compare
         the modeled results vs the real results"
    """
    def predictSupportVectorMachine(self, coefficients):
        from . import MortrackML_Library as mSL
        numberOfRows = len(self.x_samplesList)
        svcCoefficients = coefficients
        # We get the predicted data with the trained Kernel Support Vector
        # Classification (K-SVC) model
        predictedData = []
        numberOfCoefficients = len(svcCoefficients)
        for row in range(0, numberOfRows):
            temporalRow = []
            wx = 0
            for column in range(0, numberOfCoefficients-1):
                wx = wx + (self.x_samplesList[row][column])*svcCoefficients[column+1][0]
            c = -svcCoefficients[0][0] # c=ln(y=0)-b0
            
            if (wx >= c):
                temporalRow.append(1) # Its a positive sample
            else:
                temporalRow.append(-1) # Its a negative sample
            predictedData.append(temporalRow)
            
        # We return the predicted data
        return predictedData
        
    """
    predictKernelSupportVectorMachine(coefficients="We give the kernel and the SVC mathematical coefficients that we want to predict with",
                                      isPolynomialSVC="True if you want to apply a polynomial SVC. False if otherwise is desired",
                                      orderOfPolynomialSVC="If you apply a polynomial SVC through the argument isPolynomialSVC, you then give a whole number here to indicate the order of degree that you desire in such Polynomial SVC",
                                      orderOfPolynomialKernel="if you selected polynomial kernel in the kernel argument, you then here give a whole number to indicate the order of degree that you desire in such Polynomial Kernel",
                                      kernel="you specify here the type of kernel that you want to predict with. literally write, in strings, gaussian for a gaussian kernel; polynomial for a polynomial kernel; and linear for a linear kernel")
    
    This method returns the predicting values of the independent input values
    that you assign in the local variable of this class: "self.x_samplesList".
    The prediction will be made accordingly to the coefficients and
    configuration specified in the arguments of this method.
    
    CODE EXAMPLE:
        x = [
             [2, 3],
             [3, 2],
             [4, 3],
             [3, 4],
             [1, 3],
             [3, 1],
             [5, 3],
             [3, 5],
             [3, 3]
             ]
        y = [
             [1],
             [1],
             [1],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0]
             ]
        
        matrix_y = []
        for row in range(0, len(y)):
            temporalRow = []
            if (y[row][0] == 0):
                temporalRow.append(-1)
            if (y[row][0] == 1):
                temporalRow.append(1)
            if ((y[row][0]!=0) and (y[row][0]!=1)):
                raise Exception('ERROR: The dependent variable y has values different from 0 and 1.')
            matrix_y.append(temporalRow)
        matrix_x = []
        for row in range(0, len(y)):
            temporalRow = []
            for column in range(0, len(x[0])):
                temporalRow.append(x[row][column])
            matrix_x.append(temporalRow)
        
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        classification = mSL.Classification(matrix_x, matrix_y)
        # evtfbmip stands for "Eliminate Variables To Find Better Model If Possible"
        modelingResults = classification.getKernelSupportVectorMachine(kernel='gaussian', isPolynomialSVC=True, orderOfPolynomialSVC=2, orderOfPolynomialKernel=3, evtfbmip=True)
        modelCoefficients = modelingResults[0]
        acurracy = modelingResults[1]
        predictedData = modelingResults[2]
        coefficientDistribution = modelingResults[3]
        allModeledAccuracies = modelingResults[4]
        
        # --------------------------------------------------------------------------- #
        # ----- WE VISUALIZE OUR RESULTS: CBES WHEN VOLTAGE APPLIED IS POSITIVE ----- #
        # --------------------------------------------------------------------------- #
        # Visualising the Training set results
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import ListedColormap
        plt.figure()
            
        
        # We plot the Background
        x1_samples = []
        x2_samples = []
        for row in range(0, len(matrix_x)):
            x1_samples.append(matrix_x[row][0])
            x2_samples.append(matrix_x[row][1])
        # linspace(start, stop, num=50)
        x1_distance = min(x1_samples) - max(x1_samples)
        x2_distance = min(x2_samples) - max(x2_samples)
        x1_background = np.linspace(min(x1_samples)+x1_distance*0.1, max(x1_samples)-x1_distance*0.1, num=100)
        x2_background = np.linspace(min(x2_samples)+x2_distance*0.1, max(x2_samples)-x2_distance*0.1, num=100)
        predictThisValues = []
        for row in range(0, len(x1_background)):
            for row2 in range(0, len(x2_background)):
                temporalRow = []
                temporalRow.append(x1_background[row])
                temporalRow.append(x2_background[row2])
                predictThisValues.append(temporalRow)
        classification.set_xSamplesList(predictThisValues)
        predictedValuesForBg = classification.predictKernelSupportVectorMachine(coefficients=modelCoefficients, isPolynomialSVC=True, orderOfPolynomialSVC=2, orderOfPolynomialKernel=3, kernel='gaussian')
        positives_x = []
        positives_y = []
        negatives_x = []
        negatives_y = []
        for row in range(0, len(predictedValuesForBg)):
            temporalRow = []
            if (predictedValuesForBg[row][0] == 1):
                temporalRow = []
                temporalRow.append(predictThisValues[row][1])
                positives_y.append(temporalRow)
                temporalRow = []
                temporalRow.append(predictThisValues[row][0])
                positives_x.append(temporalRow)
            else:
                temporalRow = []
                temporalRow.append(predictThisValues[row][1])
                negatives_y.append(temporalRow)
                temporalRow = []
                temporalRow.append(predictThisValues[row][0])
                negatives_x.append(temporalRow)
        plt.scatter(positives_x, positives_y, c='green', s=10, label='predicted positives (1)', alpha = 0.1)
        plt.scatter(negatives_x, negatives_y, c='red', s=10, label='predicted negatives (-1)', alpha = 0.1)
        
        
        # We plot the predicted values of our currently trained model
        positives_x = []
        positives_y = []
        negatives_x = []
        negatives_y = []
        for row in range(0, len(matrix_y)):
            temporalRow = []
            if (matrix_y[row][0] == 1):
                temporalRow = []
                temporalRow.append(matrix_x[row][1])
                positives_y.append(temporalRow)
                temporalRow = []
                temporalRow.append(matrix_x[row][0])
                positives_x.append(temporalRow)
            else:
                temporalRow = []
                temporalRow.append(matrix_x[row][1])
                negatives_y.append(temporalRow)
                temporalRow = []
                temporalRow.append(matrix_x[row][0])
                negatives_x.append(temporalRow)
        plt.scatter(positives_x, positives_y, c='green', s=50, label='real positives (1)')
        plt.scatter(negatives_x, negatives_y, c='red', s=50, label='real negatives (-1)')
        # Finnally, we define the desired title, the labels and the legend for the data
        # points
        plt.title('Real Results')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.grid()
        # We show the graph with all the specifications we just declared.
        plt.show()
        
        
    EXPECTED CODE RESULT:
        "A graph will pop and will show the predicted region of the obtained
         model and the scattered points of the true/real results to compare
         the modeled results vs the real results"
    """
    def predictKernelSupportVectorMachine(self, coefficients, isPolynomialSVC=True, orderOfPolynomialSVC=3, orderOfPolynomialKernel=3, kernel='gaussian'):
        if ((kernel!='linear') and (kernel!='polynomial') and (kernel!='gaussian')):
            raise Exception('ERROR: The selected Kernel does not exist or has not been programmed in this method yet.')
        from . import MortrackML_Library as mSL
        import math
        numberOfRows = len(self.x_samplesList)
        # We create the local variables needed to run this algorithm
        kernelCoefficients = coefficients[0]
        svcCoefficients = coefficients[1]
            
        if (kernel=='gaussian'):
            # We obtain the coordinates of only the new dimentional space created
            # by the obtained kernel
            kernelData = []
            numberOfCoefficients = len(kernelCoefficients)
            gaussAproximationOrder = 2
            for row in range(0, numberOfRows):
                temporalRow = []
                actualIc = kernelCoefficients[0][0]
                currentOrderOfThePolynomial = 1
                currentVariable = 0
                for currentIndependentVariable in range(0, numberOfCoefficients-1):
                    if (currentOrderOfThePolynomial == (gaussAproximationOrder+1)):
                        currentOrderOfThePolynomial = 1
                        currentVariable = currentVariable + 1
                    actualIc = actualIc + kernelCoefficients[currentIndependentVariable+1][0]*self.x_samplesList[row][currentVariable]**(currentOrderOfThePolynomial)
                    currentOrderOfThePolynomial = currentOrderOfThePolynomial + 1
                temporalRow.append(math.exp(-actualIc))
                kernelData.append(temporalRow)
            
            # We create the new matrix of the independent variables (x) but with
            # the new dimentional space distortion made by the Kernel created
            newMatrix_x = []
            for row in range(0, numberOfRows):
                temporalRow = []
                for column in range(0, len(self.x_samplesList[0])):
                    temporalRow.append(self.x_samplesList[row][column])
                temporalRow.append(kernelData[row][0])
                newMatrix_x.append(temporalRow)
            
            # We get the predicted data with the trained Kernel Support Vector
            # Classification (K-SVC) model
            predictedData = []
            numberOfCoefficients = len(svcCoefficients)
            for row in range(0, numberOfRows):
                temporalRow = []
                wx = 0
                if (isPolynomialSVC==True):
                    currentOrderOfThePolynomial = 1
                    currentVariable = 0
                    for currentIndependentVariable in range(0, numberOfCoefficients-1):    
                        if (currentOrderOfThePolynomial == (orderOfPolynomialSVC+1)):
                            currentOrderOfThePolynomial = 1
                            currentVariable = currentVariable + 1
                        wx = wx + svcCoefficients[currentIndependentVariable+1][0]*newMatrix_x[row][currentVariable]**(currentOrderOfThePolynomial)
                        currentOrderOfThePolynomial = currentOrderOfThePolynomial + 1
                else:
                    for column in range(0, numberOfCoefficients-1):
                        wx = wx + (newMatrix_x[row][column])*svcCoefficients[column+1][0]
                c = -svcCoefficients[0][0] # c=ln(y=0)-b0
                
                if (wx >= c):
                    temporalRow.append(1) # Its a positive sample
                else:
                    temporalRow.append(-1) # Its a negative sample
                predictedData.append(temporalRow)
                
            # We return the predicted data
            return predictedData
        
        if (kernel=='polynomial'):
            # We obtain the coordinates of only the new dimentional space created
            # by the obtained kernel
            kernelData = []
            numberOfCoefficients = len(kernelCoefficients)
            for row in range(0, numberOfRows):
                temporalRow = []
                actualIc = kernelCoefficients[0][0]
                currentOrderOfThePolynomial = 1
                currentVariable = 0
                for currentIndependentVariable in range(0, numberOfCoefficients-1):
                    if (currentOrderOfThePolynomial == (orderOfPolynomialKernel+1)):
                        currentOrderOfThePolynomial = 1
                        currentVariable = currentVariable + 1
                    actualIc = actualIc + kernelCoefficients[currentIndependentVariable+1][0]*self.x_samplesList[row][currentVariable]**(currentOrderOfThePolynomial)
                    currentOrderOfThePolynomial = currentOrderOfThePolynomial + 1
                temporalRow.append(actualIc)
                kernelData.append(temporalRow)
                
            # We create the new matrix of the independent variables (x) but with
            # the new dimentional space distortion made by the Kernel created
            newMatrix_x = []
            for row in range(0, numberOfRows):
                temporalRow = []
                for column in range(0, len(self.x_samplesList[0])):
                    temporalRow.append(self.x_samplesList[row][column])
                temporalRow.append(kernelData[row][0])
                newMatrix_x.append(temporalRow)
            # We get the predicted data with the trained Kernel Support Vector
            # Classification (K-SVC) model
            predictedData = []
            numberOfCoefficients = len(svcCoefficients)
            for row in range(0, numberOfRows):
                temporalRow = []
                wx = 0
                if (isPolynomialSVC==True):
                    currentOrderOfThePolynomial = 1
                    currentVariable = 0
                    for currentIndependentVariable in range(0, numberOfCoefficients-1):    
                        if (currentOrderOfThePolynomial == (orderOfPolynomialSVC+1)):
                            currentOrderOfThePolynomial = 1
                            currentVariable = currentVariable + 1
                        wx = wx + svcCoefficients[currentIndependentVariable+1][0]*newMatrix_x[row][currentVariable]**(currentOrderOfThePolynomial)
                        currentOrderOfThePolynomial = currentOrderOfThePolynomial + 1
                else:
                    for column in range(0, numberOfCoefficients-1):
                        wx = wx + (newMatrix_x[row][column])*svcCoefficients[column+1][0]
                c = -svcCoefficients[0][0] # c=ln(y=0)-b0
                
                if (wx >= c):
                    temporalRow.append(1) # Its a positive sample
                else:
                    temporalRow.append(-1) # Its a negative sample
                predictedData.append(temporalRow)
                
            # We return the predicted data
            return predictedData

        if (kernel=='linear'):
            # We obtain the coordinates of only the new dimentional space created
            # by the obtained kernel
            kernelData = []
            numberOfIndependentVariables = len(self.x_samplesList[0])
            for row in range(0, numberOfRows):
                temporalRow = []
                actualIc = kernelCoefficients[0][0]
                for currentIndependentVariable in range(0, numberOfIndependentVariables):
                    actualIc = actualIc + kernelCoefficients[currentIndependentVariable+1][0]*self.x_samplesList[row][currentIndependentVariable]
                temporalRow.append(actualIc)
                kernelData.append(temporalRow)
                
            # We create the new matrix of the independent variables (x) but with
            # the new dimentional space distortion made by the Kernel created
            newMatrix_x = []
            for row in range(0, numberOfRows):
                temporalRow = []
                for column in range(0, len(self.x_samplesList[0])):
                    temporalRow.append(self.x_samplesList[row][column])
                temporalRow.append(kernelData[row][0])
                newMatrix_x.append(temporalRow)
            # We get the predicted data with the trained Kernel Support Vector
            # Classification (K-SVC) model
            predictedData = []
            numberOfCoefficients = len(svcCoefficients)
            for row in range(0, numberOfRows):
                temporalRow = []
                wx = 0
                if (isPolynomialSVC==True):
                    currentOrderOfThePolynomial = 1
                    currentVariable = 0
                    for currentIndependentVariable in range(0, numberOfCoefficients-1):    
                        if (currentOrderOfThePolynomial == (orderOfPolynomialSVC+1)):
                            currentOrderOfThePolynomial = 1
                            currentVariable = currentVariable + 1
                        wx = wx + svcCoefficients[currentIndependentVariable+1][0]*newMatrix_x[row][currentVariable]**(currentOrderOfThePolynomial)
                        currentOrderOfThePolynomial = currentOrderOfThePolynomial + 1
                else:
                    for column in range(0, numberOfCoefficients-1):
                        wx = wx + (newMatrix_x[row][column])*svcCoefficients[column+1][0]
                c = -svcCoefficients[0][0] # c=ln(y=0)-b0
                
                if (wx >= c):
                    temporalRow.append(1) # Its a positive sample
                else:
                    temporalRow.append(-1) # Its a negative sample
                predictedData.append(temporalRow)
                
            # We return the predicted data
            return predictedData
        
    """
    predictLinearLogisticClassifier(coefficients="We give the Linear Logistic mathematical coefficients that we want to predict with",
                                    threshold="We give a value from 0 to 1 to indicate the threshold that we want to apply to classify the predicted data with the Linear Logistic Classifier")
    
    This method returns the predicting values of the independent input values
    that you assign in the local variable of this class: "self.x_samplesList".
    The prediction will be made accordingly to the coefficients and
    configuration specified in the arguments of this method.
    
    CODE EXAMPLE:
        matrix_x = [
                [0,2],
                [1,3],
                [2,4],
                [3,5],
                [4,6],
                [5,7],
                [6,8],
                [7,9],
                [8,10],
                [9,11]
                ]
        
        matrix_y = [
                [0],
                [0],
                [1],
                [0],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1]
                ]
        
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        regression = mSL.Regression(matrix_x, matrix_y)
        # evtfbmip stands for "Eliminate Variables To Find Better Model If Possible"
        modelingResults = regression.getLinearLogisticRegression(evtfbmip=True)
        modelCoefficients = modelingResults[0]
        acurracy = modelingResults[1]
        predictedData = modelingResults[2]
        coefficientDistribution = modelingResults[3]
        allModeledAccuracies = modelingResults[4]
        
        # --------------------------------------------------------------------------- #
        # ----- WE VISUALIZE OUR RESULTS: CBES WHEN VOLTAGE APPLIED IS POSITIVE ----- #
        # --------------------------------------------------------------------------- #
        # Visualising the Training set results
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import ListedColormap
        plt.figure()
            
        
        # We plot the Background
        x1_samples = []
        x2_samples = []
        for row in range(0, len(matrix_x)):
            x1_samples.append(matrix_x[row][0])
            x2_samples.append(matrix_x[row][1])
        # linspace(start, stop, num=50)
        x1_distance = min(x1_samples) - max(x1_samples)
        x2_distance = min(x2_samples) - max(x2_samples)
        x1_background = np.linspace(min(x1_samples)+x1_distance*0.1, max(x1_samples)-x1_distance*0.1, num=100)
        x2_background = np.linspace(min(x2_samples)+x2_distance*0.1, max(x2_samples)-x2_distance*0.1, num=100)
        predictThisValues = []
        for row in range(0, len(x1_background)):
            for row2 in range(0, len(x2_background)):
                temporalRow = []
                temporalRow.append(x1_background[row])
                temporalRow.append(x2_background[row2])
                predictThisValues.append(temporalRow)
        classification = mSL.Classification(predictThisValues, [])
        predictedValuesForBg = classification.predictLinearLogisticClassifier(coefficients=modelCoefficients, threshold=0.5)
        positives_x = []
        positives_y = []
        negatives_x = []
        negatives_y = []
        for row in range(0, len(predictedValuesForBg)):
            temporalRow = []
            if (predictedValuesForBg[row][0] == 1):
                temporalRow = []
                temporalRow.append(predictThisValues[row][1])
                positives_y.append(temporalRow)
                temporalRow = []
                temporalRow.append(predictThisValues[row][0])
                positives_x.append(temporalRow)
            else:
                temporalRow = []
                temporalRow.append(predictThisValues[row][1])
                negatives_y.append(temporalRow)
                temporalRow = []
                temporalRow.append(predictThisValues[row][0])
                negatives_x.append(temporalRow)
        plt.scatter(positives_x, positives_y, c='green', s=10, label='predicted positives (1)', alpha = 0.1)
        plt.scatter(negatives_x, negatives_y, c='red', s=10, label='predicted negatives (-1)', alpha = 0.1)
        
        
        # We plot the predicted values of our currently trained model
        positives_x = []
        positives_y = []
        negatives_x = []
        negatives_y = []
        for row in range(0, len(matrix_y)):
            temporalRow = []
            if (matrix_y[row][0] == 1):
                temporalRow = []
                temporalRow.append(matrix_x[row][1])
                positives_y.append(temporalRow)
                temporalRow = []
                temporalRow.append(matrix_x[row][0])
                positives_x.append(temporalRow)
            else:
                temporalRow = []
                temporalRow.append(matrix_x[row][1])
                negatives_y.append(temporalRow)
                temporalRow = []
                temporalRow.append(matrix_x[row][0])
                negatives_x.append(temporalRow)
        plt.scatter(positives_x, positives_y, c='green', s=50, label='real positives (1)')
        plt.scatter(negatives_x, negatives_y, c='red', s=50, label='real negatives (-1)')
        # Finnally, we define the desired title, the labels and the legend for the data
        # points
        plt.title('Real Results')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.grid()
        # We show the graph with all the specifications we just declared.
        plt.show()
        
        
    EXPECTED CODE RESULT:
        "A graph will pop and will show the predicted region of the obtained
         model and the scattered points of the true/real results to compare
         the modeled results vs the real results"
    """
    def predictLinearLogisticClassifier(self, coefficients, threshold):
        from . import MortrackML_Library as mSL
        import math
        numberOfRows = len(self.x_samplesList)
        # We determine the accuracy of the obtained coefficientsfor the 
        # Probability Equation of the Logistic Regression Equation
        predictedData = []
        numberOfIndependentVariables = len(self.x_samplesList[0])
        for row in range(0, numberOfRows):
            temporalRow = []
            actualIc = coefficients[0][0]
            for currentIndependentVariable in range(0, numberOfIndependentVariables):
                actualIc = actualIc + coefficients[currentIndependentVariable+1][0]*self.x_samplesList[row][currentIndependentVariable]
            actualIc = math.exp(actualIc)
            actualIc = actualIc/(1+actualIc)
            temporalRow.append(actualIc)
            predictedData.append(temporalRow)
        
        predictedDataWithThreshold = []
        for row in range(0, numberOfRows):
            temporalRow = []
            if (predictedData[row][0]>=threshold):
                temporalRow.append(1)
            else:
                temporalRow.append(0)
            predictedDataWithThreshold.append(temporalRow)
        
        # We return the predicted data
        return predictedDataWithThreshold
        
        
"""
The ReinforcementLearning Class gives several methods to make a model that is
able to learn in real time to predict the best option among the ones you tell
it it has available. This is very useful when you actually dont have a dataset
to tell your model the expected output values to compare them and train itself
with them.

Regression("independent values (x) or options that your model will have available to pick from")
"""    
class ReinforcementLearning:
    def __init__(self, y_samplesList):
        self.y_samplesList = y_samplesList
    
    def set_ySamplesList(self, y_samplesList):
        self.y_samplesList = y_samplesList
        
    """
    getUpperConfidenceBound()
    
    This method helps you to identify what is the best option (these are called
    as arms in this algorithm) among many, to get the best number of successful
    results when theres actually no possible way to know anything about a
    particular problem that we want to figure out how to solve.
    Unlike the normal method "getRealTimeUpperConfidenceBound()", this method
    cannot solve a problem in real time, since it needs that you already have
    meassured several rounds so that then this algorithm studies it to then
    tell you which arm is the best option among all the others.
    
    This methods advantages:
            * When this algorithm tries to identify the best arm, it only needs
              to know if his current selection was successful or not (0 or 1)
              and it doesnt need to know, in that round, anything about the
              other arms
    This methods disadvantages:
            * This is the method that takes the most time to be able to
              identify the best arm. Just so that you have it in mind, for a
              problem to solve, this algorithm needed around the following
              round samples to start identifying the best arm / option for a
              random problem that i wanted to solve:
                  + For 2 arms --> around 950 samples
                  + For 3 arms --> around 1400 samples
                  + For 4 arms --> around 1200 samples
                  + For 5 arms --> around 320 samples
                  + For 6 arms --> around 350 samples
                  + For 7 arms --> around 400 samples
                  + For 8 arms --> around 270 samples
                  + For 9 arms --> around 600 samples
                  + For 10 arms --> around 600 samples
              As you can see, there is clearly no proportionality alone by the
              number of available arms and it is most likely that the needed
              number of samples, so that this algorithm starts identifying the
              best arm, will most likely depend on the probability of occurence
              for each option available to be selected by the algorithm. This
              is a great deficit for this algorithm since according to the
              situations were we are supposed to need this algorithm, we are
              supposed to not know such probability of occurence.
              
    NOTE: The logic of this algorithm follows the one described and teached by
    the Machine Learning Course "Machine Learning A-Z™: Hands-On Python & R In
    Data Science" teached by " Kirill Eremenko, Hadelin de Ponteves,
    SuperDataScience Team, SuperDataScience Support". I mention this because i
    dont quite agree with how this algorithm works but, even though i havent
    checked, there is a great chance that this is how other data scientists do
    Upper Confidence Bound.
    
    CODE EXAMPLE:
        import pandas as pd
        dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
           
        matrix_y = []
        for row in range(0, len(dataset)):
            temporalRow = []
            for column in range(0, len(dataset.iloc[0])):
                temporalRow.append(dataset.iloc[row,column])
            matrix_y.append(temporalRow)
        
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        rL = mSL.ReinforcementLearning(matrix_y)
        modelingResults = rL.getUpperConfidenceBound()
        acurracy = modelingResults[1]
        historyOfPredictedData = modelingResults[3]
            
        # ------------------------------------ #
        # ----- WE VISUALIZE OUR RESULTS ----- #
        # ------------------------------------ #
        import matplotlib.pyplot as plt
        import numpy as np
        histogram_x_data = []
        for row in range(0, len(historyOfPredictedData)):
            histogram_x_data.append(historyOfPredictedData[row][0])
        plt.figure()
        plt.hist(histogram_x_data)
        plt.title('Histogram of ads selections by UCB model')
        plt.xlabel('Ads')
        plt.ylabel('Number of times each ad was selected')
        plt.show()
        
        
    EXPECTED CODE RESULT:
        "A histogram graph will pop and will show the number of times that the
        algorithm picked each of the available options. The option with the
        highest number of selections by the algorithm is basically going to be
        the best option among them all"
        
        acurracy =
        21.78
        
        historyOfPredictedData =
        NOTE: We wont show this result because it has 10'000 rows and its just
        way too long to show here as a demonstration.
    """
    def getUpperConfidenceBound(self):
        import math
        numberOfSamples = len(self.y_samplesList)
        numberOfOptionsAvailable = len(self.y_samplesList[0])
        adsSelectedByTheAlgorithm = []
        numberOfSelectionsOfAds = []
        temporalRow = []
        for column in range(0, numberOfOptionsAvailable):
            temporalRow.append(0)
        numberOfSelectionsOfAds.append(temporalRow)
        sumsOfRewardsForEachAd = []
        temporalRow = []
        for column in range(0, numberOfOptionsAvailable):
            temporalRow.append(0)
        sumsOfRewardsForEachAd.append(temporalRow)
        totalRewards = 0
        currentUpperConfidenceBound = 0
        meanOfRewards = 0
        for row in range(0, numberOfSamples):
            highestUpperConfidenceBound = 0
            currentAdSelected = 0
            for column in range(0, numberOfOptionsAvailable):
                if (numberOfSelectionsOfAds[0][column] > 0):
                    meanOfRewards = sumsOfRewardsForEachAd[0][column] / numberOfSelectionsOfAds[0][column]
                    delta_i = math.sqrt(3/2 * math.log(row+1) / numberOfSelectionsOfAds[0][column])
                    currentUpperConfidenceBound = meanOfRewards + delta_i
                else:
                    currentUpperConfidenceBound = 1e400 # the idea is to assign a very big value to this variable
                if (currentUpperConfidenceBound > highestUpperConfidenceBound):
                    highestUpperConfidenceBound = currentUpperConfidenceBound
                    currentAdSelected = column
            temporalRow = []
            temporalRow.append(currentAdSelected)
            adsSelectedByTheAlgorithm.append(temporalRow)
            numberOfSelectionsOfAds[0][currentAdSelected] = numberOfSelectionsOfAds[0][currentAdSelected] + 1
            currentReward = self.y_samplesList[row][currentAdSelected]  
            sumsOfRewardsForEachAd[0][currentAdSelected] = sumsOfRewardsForEachAd[0][currentAdSelected] + currentReward
            totalRewards = totalRewards + currentReward
        
        accuracy = 100*totalRewards/numberOfSamples
        modelingResults = []
        modelingResults.append(0) # Null value since this model doesnt give coefficients at all
        modelingResults.append(accuracy)
        modelingResults.append(sumsOfRewardsForEachAd)
        modelingResults.append(adsSelectedByTheAlgorithm)
        return modelingResults
    
    """
    getRealTimeUpperConfidenceBound(currentNumberOfSamples="You have to indicate here the current number of samples that have occured for a particular UCB problem to solve",
                                    sumsOfRewardsForEachArm="You have to indicate here the sums of rewards for each of the available arms for a particular UCB problem to solve",
                                    numberOfSelectionsOfArms="You have to indicate here the number of times that each arm was selected by the algorithm for a particular UCB problem to solve")
    
    IMPORTANT NOTE: WHEN YOU RUN THIS METHOD TO SOLVE THE VERY FIRST ROUND OF A
                    PARTICULAR UCB PROBLEM, DONT DEFINE ANY VALUES IN THE
                    ARGUMENTS OF THIS METHOD. FOR FURTHER ROUNDS, INPUT IN THE
                    ARGUMENTS THE OUTPUT VALUES OF THE LAST TIME YOU RAN THIS
                    METHOD (SEE CODE EXAMPLE).
    
    This method helps you to identify what is the best option (these are called
    as arms in this algorithm) among many, to get the best number of successful
    results when theres actually no possible way to know anything about a
    particular problem that we want to figure out how to solve.
    Unlike the normal method "getUpperConfidenceBound()", this method learns in
    real time, while "getUpperConfidenceBound()" expects you to already have
    measured several rounds.
    
    This methods advantages:
            * When this algorithm tries to identify the best arm, it only needs
              to know if his current selection was successful or not (0 or 1)
              and it doesnt need to know, in that round, anything about the
              other arms
    This methods disadvantages:
            * This is the method that takes the most time to be able to
              identify the best arm. Just so that you have it in mind, for a
              problem to solve, this algorithm needed around the following
              round samples to start identifying the best arm / option for a
              random problem that i wanted to solve:
                  + For 2 arms --> around 950 samples
                  + For 3 arms --> around 1400 samples
                  + For 4 arms --> around 1200 samples
                  + For 5 arms --> around 320 samples
                  + For 6 arms --> around 350 samples
                  + For 7 arms --> around 400 samples
                  + For 8 arms --> around 270 samples
                  + For 9 arms --> around 600 samples
                  + For 10 arms --> around 600 samples
              As you can see, there is clearly no proportionality alone by the
              number of available arms and it is most likely that the needed
              number of samples, so that this algorithm starts identifying the
              best arm, will most likely depend on the probability of occurence
              for each option available to be selected by the algorithm. This
              is a great deficit for this algorithm since according to the
              situations were we are supposed to need this algorithm, we are
              supposed to not know such probability of occurence.
              
    NOTE: The logic of this algorithm follows the one described and teached by
    the Machine Learning Course "Machine Learning A-Z™: Hands-On Python & R In
    Data Science" teached by " Kirill Eremenko, Hadelin de Ponteves,
    SuperDataScience Team, SuperDataScience Support". I mention this because i
    dont quite agree with how this algorithm works but, even though i havent
    checked, there is a great chance that this is how other data scientists do
    Upper Confidence Bound.
    
    CODE EXAMPLE:
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        import pandas as pd
        dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
        matrix_y = []
        for row in range(0, len(dataset)):
            temporalRow = []
            for column in range(0, len(dataset.iloc[0])):
                temporalRow.append(dataset.iloc[row,column])
            matrix_y.append(temporalRow)
        
        # With this for-loop, we will simulate that we are getting the data in
        # real-time and that we are, at the same time, giving it to the algorithm
        numberOfArmsAvailable = len(matrix_y[0])
        for currentSample in range(0, len(matrix_y)):
            rL = mSL.ReinforcementLearning([matrix_y[currentSample]])
            if (currentSample == 0):
                modelingResults = rL.getRealTimeUpperConfidenceBound()
            else:
                modelingResults = rL.getRealTimeUpperConfidenceBound(currentNumberOfSamples, sumsOfRewardsForEachArm, numberOfSelectionsOfArms)
            currentNumberOfSamples = modelingResults[0]
            currentTotalAccuracy = modelingResults[1]
            sumsOfRewardsForEachArm = modelingResults[2]
            numberOfSelectionsOfArms = modelingResults[3]
            
        # ------------------------------------ #
        # ----- WE VISUALIZE OUR RESULTS ----- #
        # ------------------------------------ #
        import matplotlib.pyplot as plt
        import numpy as np
        histogram_x_data = []
        # We now add the real selected options by the algorithm
        for currentArm in range(0, numberOfArmsAvailable):
            for selectedTimes in range(0, numberOfSelectionsOfArms[0][currentArm]):
                histogram_x_data.append(currentArm)
        plt.figure()
        plt.hist(histogram_x_data)
        plt.title('Histogram of ads selections by UCB model')
        plt.xlabel('Ads')
        plt.ylabel('Number of times each ad was selected')
        plt.show()
        
        
    EXPECTED CODE RESULT:
        "A histogram graph will pop and will show the number of times that the
        algorithm picked each of the available options. The option with the
        highest number of selections by the algorithm is basically going to be
        the best option among them all"
        
        currentNumberOfSamples=
        10000
        
        currentTotalAccuracy =
        21.78
        
        sumsOfRewardsForEachArm =
        [[120, 47, 7, 38, 1675, 1, 27, 236, 20, 7]]
        
        numberOfSelectionsOfArms =
        [[705, 387, 186, 345, 6323, 150, 292, 1170, 256, 186]]
    """
    def getRealTimeUpperConfidenceBound(self, currentNumberOfSamples=0, sumsOfRewardsForEachArm=[], numberOfSelectionsOfArms=[]):
        import math
        # We save on this variable the number of arms (options) available
        numberOfArmsAvailable = len(self.y_samplesList[0])
        # We innitialize the variables that have to be innitialized only the
        # first time that this algorithm is ran for a particular problem
        if (currentNumberOfSamples == 0):
            # We save on this variable the number of times that each arm was picked
            # by our algorithm
            numberOfSelectionsOfArms = []
            temporalRow = []
            for column in range(0, numberOfArmsAvailable):
                temporalRow.append(0)
            numberOfSelectionsOfArms.append(temporalRow)
            # We save on this variable the number of times that we selected the
            # right arm in a way that we keep a count of this for each arm
            sumsOfRewardsForEachArm = []
            temporalRow = []
            for column in range(0, numberOfArmsAvailable):
                temporalRow.append(0)
            sumsOfRewardsForEachArm.append(temporalRow)
        
        # We innitialize the following variables that we will be using within
        # the core process of this algorithm
        highestUpperConfidenceBound = 0
        currentAdSelected = 0
        # We increase by one the number of current samples to follow up through
        # this algorithm
        currentNumberOfSamples = currentNumberOfSamples + 1 
        for column in range(0, numberOfArmsAvailable):
            if (numberOfSelectionsOfArms[0][column] > 0):
                meanOfRewards = sumsOfRewardsForEachArm[0][column] / numberOfSelectionsOfArms[0][column]
                delta_i = math.sqrt(3/2 * math.log(currentNumberOfSamples) / numberOfSelectionsOfArms[0][column])
                currentUpperConfidenceBound = meanOfRewards + delta_i
            else:
                currentUpperConfidenceBound = 1e400 # the idea is to assign a very big value to this variable
            if (currentUpperConfidenceBound > highestUpperConfidenceBound):
                highestUpperConfidenceBound = currentUpperConfidenceBound
                currentAdSelected = column
        numberOfSelectionsOfArms[0][currentAdSelected] = numberOfSelectionsOfArms[0][currentAdSelected] + 1
        currentReward = self.y_samplesList[0][currentAdSelected]  
        sumsOfRewardsForEachArm[0][currentAdSelected] = sumsOfRewardsForEachArm[0][currentAdSelected] + currentReward
        totalRewards = 0
        for column in range(0, numberOfArmsAvailable):
            totalRewards = totalRewards + sumsOfRewardsForEachArm[0][column]
        currentTotalAccuracy = 100*totalRewards/currentNumberOfSamples
        modelingResults = []
        modelingResults.append(currentNumberOfSamples)
        modelingResults.append(currentTotalAccuracy)
        modelingResults.append(sumsOfRewardsForEachArm)
        modelingResults.append(numberOfSelectionsOfArms)
        return modelingResults
    
    """
    getModifiedUpperConfidenceBound()
    
    This method helps you to identify what is the best option (these are called
    as arms in this algorithm) among many, to get the best number of successful
    results when theres actually no possible way to know anything about a
    particular problem that we want to figure out how to solve.
    Unlike the method "getRealTimeModifiedUpperConfidenceBound()" which learns
    in real-time, this method does not and it requires that you have already
    meassured several rounds to the input them to this method.
    
    This methods advantages:
            * This method is the fastest of all, so far, to detect the best
              possible arm (option) among all the available ones:
                  + For 2 arms --> around 1 sample
                  + For 3 arms --> around 1 sample
                  + For 4 arms --> around 1 sample
                  + For 5 arms --> around 60 samples
                  + For 6 arms --> around 60 samples
                  + For 7 arms --> around 60 samples
                  + For 8 arms --> around 60 samples
                  + For 9 arms --> around 60 samples
                  + For 10 arms --> around 60 samples
              As you can see, there is clearly no proportionality alone by the
              number of available arms and it is most likely that the needed
              number of samples, so that this algorithm starts identifying the
              best arm, will most likely depend on the probability of occurence
              for each option available to be selected by the algorithm. This
              is a great deficit for this algorithm since according to the
              situations were we are supposed to need this algorithm, we are
              supposed to not know such probability of occurence. 
    This methods disadvantages:
            * When this algorithm tries to identify the best arm, it needs to
              know, for each arm (regardless of the one picked by the 
              algorithm), if they were a successful pick or not (0 or 1),
              unlike the "getUpperConfidenceBound()" which only needs
              to know if his actual pick was sucessful or not.
    
    CODE EXAMPLE:
        import pandas as pd
        dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
           
        matrix_y = []
        for row in range(0, len(dataset)):
            temporalRow = []
            for column in range(0, len(dataset.iloc[0])):
                temporalRow.append(dataset.iloc[row,column])
            matrix_y.append(temporalRow)
        
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        rL = mSL.ReinforcementLearning(matrix_y)
        modelingResults = rL.getModifiedUpperConfidenceBound()
        acurracy = modelingResults[1]
        historyOfPredictedData = modelingResults[3]
            
        # ------------------------------------ #
        # ----- WE VISUALIZE OUR RESULTS ----- #
        # ------------------------------------ #
        import matplotlib.pyplot as plt
        import numpy as np
        histogram_x_data = []
        # We first add a fake selection for each available option (arms) so that we
        # ensure that they appear in the histogram. Otherwise, if we dont do this and
        # if the algorithm never consideres one or some of the available options, it
        # will plot considering those options never existed.
        numberOfAvailableOptions = len(matrix_y[0])
        for row in range(0, numberOfAvailableOptions):
            histogram_x_data.append(row)
        # We now add the real selected options by the algorithm
        for row in range(0, len(historyOfPredictedData)):
            histogram_x_data.append(historyOfPredictedData[row][0])
        plt.figure()
        plt.hist(histogram_x_data)
        plt.title('Histogram of ads selections by UCB model')
        plt.xlabel('Ads')
        plt.ylabel('Number of times each ad was selected')
        plt.show()
        
        
    EXPECTED CODE RESULT:
        "A histogram graph will pop and will show the number of times that the
        algorithm picked each of the available options. The option with the
        highest number of selections by the algorithm is basically going to be
        the best option among them all"
        
        acurracy =
        26.93
        
        historyOfPredictedData =
        NOTE: We wont show this result because it has 10'000 rows and its just
        way too long to show here as a demonstration.
    """
    def getModifiedUpperConfidenceBound(self):
        from . import MortrackML_Library as mSL
        numberOfSamples = len(self.y_samplesList)
        numberOfOptionsAvailable = len(self.y_samplesList[0])
        adsSelectedByTheAlgorithm = []
        numberOfSelectionsOfAds = []
        temporalRow = []
        for column in range(0, numberOfOptionsAvailable):
            temporalRow.append(0)
        numberOfSelectionsOfAds.append(temporalRow)
        sumsOfRewardsForEachAd = []
        temporalRow = []
        for column in range(0, numberOfOptionsAvailable):
            temporalRow.append(0)
        sumsOfRewardsForEachAd.append(temporalRow)
        totalRewards = 0
        currentUpperConfidenceBound = 0
        
        meanList = []
        temporalRow = []
        for column in range(0, numberOfOptionsAvailable):
            temporalRow.append(0)
        meanList.append(temporalRow)
        standardDeviationList = []
        temporalRow = []
        for column in range(0, numberOfOptionsAvailable):
            temporalRow.append(0)
        standardDeviationList.append(temporalRow)
        
        # We start the modified UCB algorithm
        for row in range(0, numberOfSamples):
            highestUpperConfidenceBound = 0
            currentAdSelected = 0
            for column in range(0, numberOfOptionsAvailable):
                # We compare all the prediction intervals to then pick the best one
                transcuredNumberOfRounds = row+1
                if (transcuredNumberOfRounds > 2):
                    tD = mSL.Tdistribution(desiredTrustInterval=99.9)
                    tValue = tD.getCriticalValue(transcuredNumberOfRounds)
                    tI = mSL.TrustIntervals()
                    predictionIntervalsList = tI.getPredictionIntervals(transcuredNumberOfRounds, [[meanList[0][column]]], [[standardDeviationList[0][column]]], tValue)
                    currentUpperConfidenceBound = predictionIntervalsList[0][1]
                else:
                    currentUpperConfidenceBound = 1e400 # the idea is to assign a very big value to this variable
                if (currentUpperConfidenceBound > highestUpperConfidenceBound):
                    highestUpperConfidenceBound = currentUpperConfidenceBound
                    currentAdSelected = column
                # We update the means and the standard deviations of all the 
                # options available for this model (arms) with the latest
                # observations that were made.
                currentReward = self.y_samplesList[row][column] 
                sumsOfRewardsForEachAd[0][column] = sumsOfRewardsForEachAd[0][column] + currentReward
                if (transcuredNumberOfRounds == 1):
                    meanList[0][column] = currentReward
                    standardDeviationList[0][column] = 0
                if (transcuredNumberOfRounds == 2):
                    firstValue = meanList[0][column]
                    meanList[0][column] = sumsOfRewardsForEachAd[0][column]/transcuredNumberOfRounds
                    standardDeviationList[0][column] = (( (firstValue-meanList[0][column])**2 + (currentReward-meanList[0][column])**2 )/(2-1) )**(0.5)
                if (transcuredNumberOfRounds > 2):
                    meanList[0][column] = sumsOfRewardsForEachAd[0][column]/transcuredNumberOfRounds
                    standardDeviationList[0][column] = (( (standardDeviationList[0][column]**2)*(transcuredNumberOfRounds-1)+(currentReward-meanList[0][column])**2 )/(transcuredNumberOfRounds-1) )**(0.5)
            # We update the list of the currently selected arm and the total
            # rewards variable
            temporalRow = []
            temporalRow.append(currentAdSelected)
            adsSelectedByTheAlgorithm.append(temporalRow)
            numberOfSelectionsOfAds[0][currentAdSelected] = numberOfSelectionsOfAds[0][currentAdSelected] + 1
            currentReward = self.y_samplesList[row][currentAdSelected]
            totalRewards = totalRewards + currentReward
        # We return the model results            
        accuracy = 100*totalRewards/numberOfSamples
        modelingResults = []
        modelingResults.append(0) # Null value since this model doesnt give coefficients at all
        modelingResults.append(accuracy)
        modelingResults.append(sumsOfRewardsForEachAd)
        modelingResults.append(adsSelectedByTheAlgorithm)
        return modelingResults
        
    """
    getRealTimeModifiedUpperConfidenceBound(currentNumberOfSamples="You have to indicate here the current number of samples that have occured for a particular UCB problem to solve",
                                    sumsOfRewardsForEachSelectedArm="You have to indicate the sums of the rewards for each arm but only for those situations were the algorithm picked each arm",
                                    numberOfSelectionsOfArms="You have to indicate here the number of times that each arm was selected by the algorithm for a particular UCB problem to solve",
                                    trueSumsOfRewardsForEachArm="You have to indicate the real number of times that each arm has been a successful result, regardless of what the algorithm identified",
                                    meanList="You have to indicate the mean list of the rewards obtained for each arm",
                                    standardDeviationList="You have to indicate the standard deviation list of the rewards obtained for each arm")
    
    IMPORTANT NOTE: WHEN YOU RUN THIS METHOD TO SOLVE THE VERY FIRST ROUND OF A
                    PARTICULAR UCB PROBLEM, DONT DEFINE ANY VALUES IN THE
                    ARGUMENTS OF THIS METHOD. FOR FURTHER ROUNDS, INPUT IN THE
                    ARGUMENTS THE OUTPUT VALUES OF THE LAST TIME YOU RAN THIS
                    METHOD (SEE CODE EXAMPLE).
    
    This method helps you to identify what is the best option (these are called
    as arms in this algorithm) among many, to get the best number of successful
    results when theres actually no possible way to know anything about a
    particular problem that we want to figure out how to solve.
    Unlike the normal method "getModifiedUpperConfidenceBound()", this method
    learns in real time, while "getModifiedUpperConfidenceBound()" expects you
    to already have measured several rounds.
    
    This methods advantages:
            * This method is the fastest of all, so far, to detect the best
              possible arm (option) among all the available ones:
                  + For 2 arms --> around 1 sample
                  + For 3 arms --> around 1 sample
                  + For 4 arms --> around 1 sample
                  + For 5 arms --> around 60 samples
                  + For 6 arms --> around 60 samples
                  + For 7 arms --> around 60 samples
                  + For 8 arms --> around 60 samples
                  + For 9 arms --> around 60 samples
                  + For 10 arms --> around 60 samples
              As you can see, there is clearly no proportionality alone by the
              number of available arms and it is most likely that the needed
              number of samples, so that this algorithm starts identifying the
              best arm, will most likely depend on the probability of occurence
              for each option available to be selected by the algorithm. This
              is a great deficit for this algorithm since according to the
              situations were we are supposed to need this algorithm, we are
              supposed to not know such probability of occurence. 
    This methods disadvantages:
            * When this algorithm tries to identify the best arm, it needs to
              know, for each arm (regardless of the one picked by the 
              algorithm), if they were a successful pick or not (0 or 1),
              unlike the "getRealTimeUpperConfidenceBound()" which only needs
              to know if his actual pick was sucessful or not.
    
    CODE EXAMPLE:
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        import pandas as pd
        dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
        matrix_y = []
        for row in range(0, len(dataset)):
            temporalRow = []
            for column in range(0, len(dataset.iloc[0])):
                temporalRow.append(dataset.iloc[row,column])
            matrix_y.append(temporalRow)
        
        # With this for-loop, we will simulate that we are getting the data in
        # real-time and that we are, at the same time, giving it to the algorithm
        numberOfArmsAvailable = len(matrix_y[0])
        for currentSample in range(0, len(matrix_y)):
            rL = mSL.ReinforcementLearning([matrix_y[currentSample]])
            if (currentSample == 0):
                modelingResults = rL.getRealTimeModifiedUpperConfidenceBound()
            else:
                modelingResults = rL.getRealTimeModifiedUpperConfidenceBound(currentNumberOfSamples, sumsOfRewardsForEachSelectedArm, numberOfSelectionsOfArms, trueSumsOfRewardsForEachArm, meanList, standardDeviationList)
            currentNumberOfSamples = modelingResults[0]
            currentTotalAccuracy = modelingResults[1]
            sumsOfRewardsForEachSelectedArm = modelingResults[2]
            numberOfSelectionsOfArms = modelingResults[3]
            trueSumsOfRewardsForEachArm = modelingResults[4]
            meanList = modelingResults[5]
            standardDeviationList = modelingResults[6]
        
        # ------------------------------------ #
        # ----- WE VISUALIZE OUR RESULTS ----- #
        # ------------------------------------ #
        import matplotlib.pyplot as plt
        import numpy as np
        histogram_x_data = []
        # We first add a fake selection for each available option (arms) so that we
        # ensure that they appear in the histogram. Otherwise, if we dont do this and
        # if the algorithm never consideres one or some of the available options, it
        # will plot considering those options never existed.
        for row in range(0, numberOfArmsAvailable):
            histogram_x_data.append(row)
        # We now add the real selected options by the algorithm
        for currentArm in range(0, numberOfArmsAvailable):
            for selectedTimes in range(0, numberOfSelectionsOfArms[0][currentArm]):
                histogram_x_data.append(currentArm)
        plt.figure()
        plt.hist(histogram_x_data)
        plt.title('Histogram of ads selections by UCB model')
        plt.xlabel('Ads')
        plt.ylabel('Number of times each ad was selected')
        plt.show()
        
        
    EXPECTED CODE RESULT:
        "A histogram graph will pop and will show the number of times that the
        algorithm picked each of the available options. The option with the
        highest number of selections by the algorithm is basically going to be
        the best option among them all"
        
        currentNumberOfSamples=
        10000
        
        currentTotalAccuracy =
        26.93
        
        sumsOfRewardsForEachSelectedArm =
        [[3, 0, 0, 0, 2690, 0, 0, 0, 0, 0]]
        
        numberOfSelectionsOfArms =
        [[25, 0, 0, 0, 9975, 0, 0, 0, 0, 0]]
        
        trueSumsOfRewardsForEachArm =
        [[1703, 1295, 728, 1196, 2695, 126, 1112, 2091, 952, 489]]
        
        meanList =
        [[0.1703,
          0.1295,
          0.0728,
          0.1196,
          0.2695,
          0.0126,
          0.1112,
          0.2091,
          0.0952,
          0.0489]]
        
        standardDeviationList =
        [[1.2506502260503618,
          1.0724240984136193,
          0.7004403369435815,
          0.9286872458865242,
          1.412843221683186,
          0.3047987328938745,
          0.7525852536272276,
          1.2007787911241279,
          1.030718190027389,
          0.5406998109413704]]
    """
    def getRealTimeModifiedUpperConfidenceBound(self, currentNumberOfSamples=0, sumsOfRewardsForEachSelectedArm=[], numberOfSelectionsOfArms=[], trueSumsOfRewardsForEachArm=[], meanList=[], standardDeviationList=[]):
        from . import MortrackML_Library as mSL
        # We save on this variable the number of arms (options) available
        numberOfArmsAvailable = len(self.y_samplesList[0])
        # We innitialize the variables that have to be innitialized only the
        # first time that this algorithm is ran for a particular problem
        if (currentNumberOfSamples == 0):
            # We save on this variable the number of times that each arm was picked
            # by our algorithm
            temporalRow = []
            for column in range(0, numberOfArmsAvailable):
                temporalRow.append(0)
            numberOfSelectionsOfArms.append(temporalRow)
            # We save on this variable the number of times that the algorithm
            # selected a particular arm
            temporalRow = []
            for column in range(0, numberOfArmsAvailable):
                temporalRow.append(0)
            sumsOfRewardsForEachSelectedArm.append(temporalRow)
            # We save on this variable the number of times that each arm had
            # been the right pick or that it gave a successful result
            temporalRow = []
            for column in range(0, numberOfArmsAvailable):
                temporalRow.append(0)
            trueSumsOfRewardsForEachArm.append(temporalRow)
            # We save on this variable the mean of the results obtained for
            # each arm
            temporalRow = []
            for column in range(0, numberOfArmsAvailable):
                temporalRow.append(0)
            meanList.append(temporalRow)
            # We save on this variable the standard deviation of the results
            # obtained for each arm
            temporalRow = []
            for column in range(0, numberOfArmsAvailable):
                temporalRow.append(0)
            standardDeviationList.append(temporalRow)
        # We innitialize the following variables that we will be using within
        # the core process of this algorithm
        highestUpperConfidenceBound = 0
        currentAdSelected = 0
        # We increase by one the number of current samples to follow up through
        # this algorithm
        currentNumberOfSamples = currentNumberOfSamples + 1 
        for column in range(0, numberOfArmsAvailable):
            # We compare all the prediction intervals to then pick the best one
            if (currentNumberOfSamples > 2):
                tD = mSL.Tdistribution(desiredTrustInterval=99.9)
                tValue = tD.getCriticalValue(currentNumberOfSamples)
                tI = mSL.TrustIntervals()
                predictionIntervalsList = tI.getPredictionIntervals(currentNumberOfSamples, [[meanList[0][column]]], [[standardDeviationList[0][column]]], tValue)
                currentUpperConfidenceBound = predictionIntervalsList[0][1]
            else:
                currentUpperConfidenceBound = 1e400 # the idea is to assign a very big value to this variable
            if (currentUpperConfidenceBound > highestUpperConfidenceBound):
                highestUpperConfidenceBound = currentUpperConfidenceBound
                currentAdSelected = column
            # We update the means and the standard deviations of all the 
            # options available for this model (arms) with the latest
            # observations that were made.
            currentReward = self.y_samplesList[0][column]
            trueSumsOfRewardsForEachArm[0][column] = trueSumsOfRewardsForEachArm[0][column] + currentReward
            if (currentNumberOfSamples == 1):
                meanList[0][column] = currentReward
                standardDeviationList[0][column] = 0
            if (currentNumberOfSamples == 2):
                firstValue = meanList[0][column]
                meanList[0][column] = trueSumsOfRewardsForEachArm[0][column]/currentNumberOfSamples
                standardDeviationList[0][column] = (( (firstValue-meanList[0][column])**2 + (currentReward-meanList[0][column])**2 )/(currentNumberOfSamples-1) )**(0.5)
            if (currentNumberOfSamples > 2):
                meanList[0][column] = trueSumsOfRewardsForEachArm[0][column]/currentNumberOfSamples
                standardDeviationList[0][column] = (( (standardDeviationList[0][column]**2)*(currentNumberOfSamples-1)+(currentReward-meanList[0][column])**2 )/(currentNumberOfSamples-1) )**(0.5)
        # We update the list of the currently selected arm and the total
        # rewards variable
        numberOfSelectionsOfArms[0][currentAdSelected] = numberOfSelectionsOfArms[0][currentAdSelected] + 1
        currentReward = self.y_samplesList[0][currentAdSelected]
        sumsOfRewardsForEachSelectedArm[0][currentAdSelected] = sumsOfRewardsForEachSelectedArm[0][currentAdSelected] + currentReward
        totalRewards = 0
        for column in range(0, numberOfArmsAvailable):
            totalRewards = totalRewards + sumsOfRewardsForEachSelectedArm[0][column]
        currentTotalAccuracy = 100*totalRewards/currentNumberOfSamples
        modelingResults = []
        modelingResults.append(currentNumberOfSamples)
        modelingResults.append(currentTotalAccuracy)
        modelingResults.append(sumsOfRewardsForEachSelectedArm)
        modelingResults.append(numberOfSelectionsOfArms)
        modelingResults.append(trueSumsOfRewardsForEachArm)
        modelingResults.append(meanList)
        modelingResults.append(standardDeviationList)
        return modelingResults
    
    
"""
The DeepLearning Class gives several methods to make a model through the
concept of how a real neuron works.

DeepLearning("mean values of the x datapoints to model", "mean values of the y datapoints to model")
"""    
class DeepLearning:
    def __init__(self, x_samplesList, y_samplesList):
        self.x_samplesList = x_samplesList
        self.y_samplesList = y_samplesList
    
    def set_xSamplesList(self, x_samplesList):
        self.x_samplesList = x_samplesList
        
    def set_ySamplesList(self, y_samplesList):
        self.y_samplesList = y_samplesList
    
    """
    getReluActivation(x="the instant independent value from which you want to know the dependent ReLU value/result")
    
    This method calculates and returns the ReLU function value of the instant
    independent value that you give in the "x" local variable of this method.
    """
    def getReluActivation(self, x):
        if (x > 0):
            return x
        else:
            return 0
        
    """
    getReluActivationDerivative(x="the instant independent value from which you want to know the derivate of the dependent ReLU value/result")
    
    This method calculates and returns the derivate ReLU function value of the
    instant independent value that you give in the "x" local variable of this
    method.
    """
    def getReluActivationDerivative(self, x):
        if (x > 0):
            return 1
        else:
            return 0
    
    """
    getTanhActivation(x="the instant independent value from which you want to know the dependent Hyperbolic Tangent (Tanh) value/result")
    
    This method calculates and returns the Hyperbolic Tangent (Tanh) function
    value of the instant independent value that you give in the "x" local
    variable of this method.
    """
    def getTanhActivation(self, x):
        import math
        a = math.exp(x)
        b = math.exp(-x)
        return ((a-b)/(a+b))
        
    """
    getReluActivation(x="the instant independent value from which you want to know the dependent Sigmoid value/result")
    
    This method calculates and returns the Sigmoid function value of the
    instant independent value that you give in the "x" local variable of this
    method.
    """
    def getSigmoidActivation(self, x):
        import math
        return (1/(1+math.exp(-x)))
    
    """
    getRaiseToTheSecondPowerActivation(x="the instant independent value from which you want to know the dependent Exponentiation value/result")
    
    This method calculates and returns the Exponentiation function value of
    the instant independent value that you give in the "x" local variable of
    this method.
    """
    def getRaiseToTheSecondPowerActivation(self, x):
        return x*x
        
    """
    getRaiseToTheSecondPowerDerivative(x="the instant independent value from which you want to know the dependent Exponentiation value/result")
    
    This method calculates and returns the derivate Exponentiation function
    value of the instant independent value that you give in the "x" local
    variable of this method.
    """
    def getRaiseToTheSecondPowerDerivative(self, x):
        return 2*x
    
    """
    getRaiseToTheThirdPowerActivation(x="the instant independent value from which you want to know the dependent Exponentiation value/result")
    
    This method calculates and returns the Exponentiation function value of
    the instant independent value that you give in the "x" local variable of
    this method.
    """
    def getRaiseToTheThirdPowerActivation(self, x):
        return x*x*x
        
    """
    getRaiseToTheThirdPowerDerivative(x="the instant independent value from which you want to know the dependent Exponentiation value/result")
    
    This method calculates and returns the derivate Exponentiation function
    value of the instant independent value that you give in the "x" local
    variable of this method.
    """
    def getRaiseToTheThirdPowerDerivative(self, x):
        return 3*x*x
    
    """
    getRaiseToTheFourthPowerActivation(x="the instant independent value from which you want to know the dependent Exponentiation value/result")
    
    This method calculates and returns the Exponentiation function value of
    the instant independent value that you give in the "x" local variable of
    this method.
    """
    def getRaiseToTheFourthPowerActivation(self, x):
        return x*x*x*x
        
    """
    getRaiseToTheFourthPowerDerivative(x="the instant independent value from which you want to know the dependent Exponentiation value/result")
    
    This method calculates and returns the derivate Exponentiation function
    value of the instant independent value that you give in the "x" local
    variable of this method.
    """
    def getRaiseToTheFourthPowerDerivative(self, x):
        return 4*x*x*x
    
    """
    getRaiseToTheFifthPowerActivation(x="the instant independent value from which you want to know the dependent Exponentiation value/result")
    
    This method calculates and returns the Exponentiation function value of
    the instant independent value that you give in the "x" local variable of
    this method.
    """
    def getRaiseToTheFifthPowerActivation(self, x):
        return x*x*x*x*x
        
    """
    getRaiseToTheFifthPowerDerivative(x="the instant independent value from which you want to know the dependent Exponentiation value/result")
    
    This method calculates and returns the derivate Exponentiation function
    value of the instant independent value that you give in the "x" local
    variable of this method.
    """
    def getRaiseToTheFifthPowerDerivative(self, x):
        return 5*x*x*x*x
    
    """
    getRaiseToTheSixthPowerActivation(x="the instant independent value from which you want to know the dependent Exponentiation value/result")
    
    This method calculates and returns the Exponentiation function value of
    the instant independent value that you give in the "x" local variable of
    this method.
    """
    def getRaiseToTheSixthPowerActivation(self, x):
        return x*x*x*x*x*x
        
    """
    getRaiseToTheSixthPowerDerivative(x="the instant independent value from which you want to know the dependent Exponentiation value/result")
    
    This method calculates and returns the derivate Exponentiation function
    value of the instant independent value that you give in the "x" local
    variable of this method.
    """
    def getRaiseToTheSixthPowerDerivative(self, x):
        return 6*x*x*x*x*x
    
    """
    getExponentialActivation(x="the instant independent value from which you want to know the dependent Exponential-Euler value/result")
    
    This method calculates and returns the Exponential-Euler function value of
    the instant independent value that you give in the "x" local variable of
    this method.
    """
    def getExponentialActivation(self, x):
        import math
        return math.exp(x)
        
    """
    getExponentialDerivative(x="the instant independent value from which you want to know the dependent Exponential-Euler value/result")
    
    This method calculates and returns the derivate Exponential-Euler function
    value of the instant independent value that you give in the "x" local
    variable of this method.
    """
    def getExponentialDerivative(self, x):
        import math
        return math.exp(x)
    
    """
    getSingleArtificialNeuron(activationFunction="the literal name, in lowercaps, of the activation function that you want to apply the neuron",
                              learningRate="the rate at which you want your neuron to learn (remember that 1=100% learning rate or normal learning rate)",
                              numberOfEpochs="The number of times you want your neuron to train itself",
                              stopTrainingIfAcurracy="define the % value that you want the neuron to stop training itself if such accuracy value is surpassed",
                              isCustomizedInitialWeights="set to True if you will define a customized innitial weight vector for each neuron. False if you want them to be generated randomly",
                              firstMatrix_w="If you set the input argument of this method isCustomizedInitialWeights to True, then assign here the customized innitial weight vectors you desire for each neuron",
                              isClassification="set to True if you are solving a classification problem. False if otherwise")
    
    This method creates a single Artificial Neuron and, within this method,
    such neuron trains itself to learn to predict the input values that it was
    given to study by comparing them with the output expected values.
    When the neuron finishes its learning process, this method will return the
    modeling results.
    
    CODE EXAMPLE:
        # matrix_y = [expectedResult]
        matrix_y = [
                [25.5],
                [31.2],
                [25.9],
                [38.4],
                [18.4],
                [26.7],
                [26.4],
                [25.9],
                [32],
                [25.2],
                [39.7],
                [35.7],
                [26.5]
                ]
        # matrix_x = [variable1, variable2, variable3]
        matrix_x = [
                [1.74, 5.3, 10.8],
                [6.32, 5.42, 9.4],
                [6.22, 8.41, 7.2],
                [10.52, 4.63, 8.5],
                [1.19, 11.6, 9.4],
                [1.22, 5.85, 9.9],
                [4.1, 6.62, 8],
                [6.32, 8.72, 9.1],
                [4.08, 4.42, 8.7],
                [4.15, 7.6, 9.2],
                [10.15, 4.83, 9.4],
                [1.72, 3.12, 7.6],
                [1.7, 5.3, 8.2]
                ]
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        dL = mSL.DeepLearning(matrix_x, matrix_y)
        modelingResults = dL.getSingleArtificialNeuron(activationFunction='none', learningRate=0.001, numberOfEpochs=100000, stopTrainingIfAcurracy=99.9, isCustomizedInitialWeights=False, firstMatrix_w=[], isClassification=False)
        modelCoefficients = modelingResults[0]
        acurracy = modelingResults[1]
        predictedData = modelingResults[2]
        firstMatrix_w = modelingResults[3]
        coefficientDistribution = modelingResults[4]
        allModeledAccuracies = modelingResults[5]
        
    RESULT OF CODE:
        modelCoefficients =
        [[28.235246103419946],
         [1.12749544645359],
         [-1.7353168202914326],
         [0.7285727543658252]]
        
        acurracy =
        95.06995458954695
        
        predictedData =
        [[28.868494779855514],
         [32.80418405006583],
         [25.89997715314427],
         [38.25484973427189],
         [16.295874460357858],
         [26.67205741761012],
         [27.198762118476985],
         [26.859066716794352],
         [31.50391014224514],
         [26.42881371215305],
         [38.14632853395502],
         [30.297502725191123],
         [26.929105800646223]]
        
        coefficientDistribution =
        "
        Coefficients distribution is as follows:
        modelCoefficients =
        [
          [Neuron1_bias, Neuron1_weight1, Neuron1_weight2, ... , Neuron1_weightM]
        ]
        "
        
        allModeledAccuracies["independent variable distribution used to get a model"]["model accuracy", "model coefficients obtained but with original distribution"] =
        # NOTE: since this variable contains large amounts of information, it
        #       will not be displayed but only described on how to use it.
    """
    def getSingleArtificialNeuron(self, activationFunction='sigmoid', learningRate=1, numberOfEpochs=1000, stopTrainingIfAcurracy=95, isCustomizedInitialWeights=False, firstMatrix_w=[], isClassification=True):
        if ((activationFunction!='none') and (activationFunction!='sigmoid') and (activationFunction!='relu') and (activationFunction!='tanh') and (activationFunction!='raiseTo2ndPower') and (activationFunction!='raiseTo3rdPower') and (activationFunction!='raiseTo4thPower') and (activationFunction!='raiseTo5thPower') and (activationFunction!='raiseTo6thPower') and (activationFunction!='exponential')):
            raise Exception('ERROR: The selected Activation Function does not exist or has not been programmed in this method yet.')
        from ..linearAlgebra import MortrackLinearAlgebraLibrary as mLAL
        # from . import MortrackML_Library as mSL
        # import math
        import random
        numberOfIndependentRows= len(self.x_samplesList)
        numberOfIndependentVariables = len(self.x_samplesList[0])
        matrix_x = []
        for row in range(0, numberOfIndependentRows):
            temporalRow = []
            temporalRow.append(1)
            for column in range(0, numberOfIndependentVariables):
                temporalRow.append(self.x_samplesList[row][column])
            matrix_x.append(temporalRow)
        matrix_y = self.y_samplesList
        # We innitialize the weight vector random values from -1 up to +1
        if (isCustomizedInitialWeights == False):
            matrix_w = []
            for row in range(0, numberOfIndependentVariables+1): # bias + w vector
                temporalRow = []
                temporalRow.append(random.random()*2-1)
                matrix_w.append(temporalRow)
            firstMatrix_w = matrix_w
        else:
            matrix_w = firstMatrix_w
        
        # We calculate the results obtained with the innitialized random weight
        # vector
        matrixMath = mLAL.MatrixMath()
        Fx = matrixMath.getDotProduct(matrix_x, matrix_w)
        Fz = []
        dFz = []
        for row in range(0, numberOfIndependentRows):
            temporalRow = []
            if (activationFunction == 'none'):
                current_Fz = Fx[row][0]
            if (activationFunction == 'sigmoid'):
                current_Fz = self.getSigmoidActivation(Fx[row][0])
            if (activationFunction == 'relu'):
                current_Fz = self.getReluActivation(Fx[row][0])
            if (activationFunction == 'tanh'):
                current_Fz = self.getTanhActivation(Fx[row][0])
            if (activationFunction == 'raiseTo2ndPower'):
                current_Fz = self.getRaiseToTheSecondPowerActivation(Fx[row][0])
            if (activationFunction == 'raiseTo3rdPower'):
                current_Fz = self.getRaiseToTheThirdPowerActivation(Fx[row][0])
            if (activationFunction == 'raiseTo4thPower'):
                current_Fz = self.getRaiseToTheFourthPowerActivation(Fx[row][0])
            if (activationFunction == 'raiseTo5thPower'):
                current_Fz = self.getRaiseToTheFifthPowerActivation(Fx[row][0])
            if (activationFunction == 'raiseTo6thPower'):
                current_Fz = self.getRaiseToTheSixthPowerActivation(Fx[row][0])
            if (activationFunction == 'exponential'):
                current_Fz = self.getExponentialActivation(Fx[row][0]) 
            temporalRow.append(current_Fz)
            Fz.append(temporalRow)
            temporalRow = []
            if (activationFunction == 'none'):
                if (current_Fz != 0):
                    temporalRow.append(1)
                else:
                    temporalRow.append(0)
            if (activationFunction == 'sigmoid'):
                temporalRow.append(current_Fz*(1-current_Fz))
            if (activationFunction == 'relu'):
                temporalRow.append(self.getReluActivationDerivative(Fx[row][0]))
            if (activationFunction == 'tanh'):
                temporalRow.append(1-current_Fz**2)
            if (activationFunction == 'raiseTo2ndPower'):
                temporalRow.append(self.getRaiseToTheSecondPowerDerivative(Fx[row][0]))
            if (activationFunction == 'raiseTo3rdPower'):
                temporalRow.append(self.getRaiseToTheThirdPowerDerivative(Fx[row][0]))
            if (activationFunction == 'raiseTo4thPower'):
                temporalRow.append(self.getRaiseToTheFourthPowerDerivative(Fx[row][0]))
            if (activationFunction == 'raiseTo5thPower'):
                temporalRow.append(self.getRaiseToTheFifthPowerDerivative(Fx[row][0]))
            if (activationFunction == 'raiseTo6thPower'):
                temporalRow.append(self.getRaiseToTheSixthPowerDerivative(Fx[row][0]))
            if (activationFunction == 'exponential'):
                temporalRow.append(self.getExponentialDerivative(Fx[row][0]))
            dFz.append(temporalRow)
        
        # We evaluate the performance of the innitialized weight vectors
        predictionAcurracy = 0
        predictedData = Fz
        numberOfDataPoints = numberOfIndependentRows
        for row in range(0, numberOfDataPoints):
            n2 = self.y_samplesList[row][0]
            n1 = predictedData[row][0]
            if (isClassification == False):
                if (((n1*n2) != 0)):
                    newAcurracyValueToAdd = (1-(abs(n2-n1)/abs(n2)))
                    if (newAcurracyValueToAdd < 0):
                        newAcurracyValueToAdd = 0
                    predictionAcurracy = predictionAcurracy + newAcurracyValueToAdd
            if (isClassification == True):
                if (abs(n1) > abs(n2)): # n2 has to be the one with the highest value with respect to n1
                    n2 = predictedData[row][0]
                    n1 = self.y_samplesList[row][0]
                if ((n1==0) and (n2>=-1 and n2<=1) and (n2!=0)):
                    predictionAcurracy = predictionAcurracy + ((1-abs(n2))/(1-n1))
                if (n1==n2):
                    predictionAcurracy = predictionAcurracy + 1
        predictionAcurracy = predictionAcurracy/numberOfDataPoints*100
        
        # We save the current the modeling results
        bestModelingResults = []
        bestModelingResults.append(matrix_w)
        bestModelingResults.append(predictionAcurracy)
        bestModelingResults.append(Fz)
        bestModelingResults.append(firstMatrix_w)
        bestModelingResults.append("Coefficients distribution is as follows:\nmodelCoefficients =\n[\n  [Neuron1_bias, Neuron1_weight1, Neuron1_weight2, ... , Neuron1_weightM]\n]\n")
        allAccuracies = []
        temporalRow = []
        temporalRow.append(bestModelingResults[1])
        temporalRow.append(bestModelingResults[0])
        allAccuracies.append(temporalRow)
        
        
        # ---------------------------------------------------------------- #
        # ----- WE START THE TRAINING PROCESS FROM EPOCH 2 AND ABOVE ----- #
        # ---------------------------------------------------------------- #
        # 2nd Epoch
        # Djtotal_Dresult = [] # = expected - predicted
        Dresult_Dsum = dFz # = dFz
        # Dsum_Dw = matrix_y # = expected result
        Djtotal_Dw = [] # = (Djtotal_Dresult)*(Dresult_Dsum)*(Dsum_Dw)
        for row in range(0, numberOfIndependentRows):
            temporalRow = []
            current_Djtotal_Dresult = matrix_y[row][0]-Fz[row][0]
            #temporalRow.append(Djtotal_Dresult[row][0]*Dresult_Dsum[row][0]*Dsum_Dw[row][0])
            temporalRow.append(current_Djtotal_Dresult*Dresult_Dsum[row][0])
            Djtotal_Dw.append(temporalRow)
        transposedMatrix_x = matrixMath.getTransposedMatrix(matrix_x)
        learningValue = matrixMath.getDotProduct(transposedMatrix_x, Djtotal_Dw)
        newMatrix_w = []
        for row in range(0, numberOfIndependentVariables+1): # bias + w vector
            temporalRow = []
            temporalRow.append(matrix_w[row][0]+learningRate*learningValue[row][0])
            newMatrix_w.append(temporalRow)
            
        # 3rd Epoch and above
        for currentEpoch in range(1, numberOfEpochs):
            print('Current Epoch = ' + format(currentEpoch))
            # ----- Predict the output values with latest weight vector ----- #
            currentMatrix_w = newMatrix_w
            Fx = matrixMath.getDotProduct(matrix_x, currentMatrix_w)
            Fz = []
            dFz = []
            for row in range(0, numberOfIndependentRows):
                temporalRow = []
                if (activationFunction == 'none'):
                    current_Fz = Fx[row][0]
                if (activationFunction == 'sigmoid'):
                    current_Fz = self.getSigmoidActivation(Fx[row][0])
                if (activationFunction == 'relu'):
                    current_Fz = self.getReluActivation(Fx[row][0])
                if (activationFunction == 'tanh'):
                    current_Fz = self.getTanhActivation(Fx[row][0])
                if (activationFunction == 'raiseTo2ndPower'):
                    current_Fz = self.getRaiseToTheSecondPowerActivation(Fx[row][0])
                if (activationFunction == 'raiseTo3rdPower'):
                    current_Fz = self.getRaiseToTheThirdPowerActivation(Fx[row][0])
                if (activationFunction == 'raiseTo4thPower'):
                    current_Fz = self.getRaiseToTheFourthPowerActivation(Fx[row][0])
                if (activationFunction == 'raiseTo5thPower'):
                    current_Fz = self.getRaiseToTheFifthPowerActivation(Fx[row][0])
                if (activationFunction == 'raiseTo6thPower'):
                    current_Fz = self.getRaiseToTheSixthPowerActivation(Fx[row][0])
                if (activationFunction == 'exponential'):
                    current_Fz = self.getExponentialActivation(Fx[row][0]) 
                temporalRow.append(current_Fz)
                Fz.append(temporalRow)
                temporalRow = []
                if (activationFunction == 'none'):
                    if (current_Fz != 0):
                        temporalRow.append(1)
                    else:
                        temporalRow.append(0)
                if (activationFunction == 'sigmoid'):
                    temporalRow.append(current_Fz*(1-current_Fz))
                if (activationFunction == 'relu'):
                    temporalRow.append(self.getReluActivationDerivative(Fx[row][0]))
                if (activationFunction == 'tanh'):
                    temporalRow.append(1-current_Fz**2)
                if (activationFunction == 'raiseTo2ndPower'):
                    temporalRow.append(self.getRaiseToTheSecondPowerDerivative(Fx[row][0]))
                if (activationFunction == 'raiseTo3rdPower'):
                    temporalRow.append(self.getRaiseToTheThirdPowerDerivative(Fx[row][0]))
                if (activationFunction == 'raiseTo4thPower'):
                    temporalRow.append(self.getRaiseToTheFourthPowerDerivative(Fx[row][0]))
                if (activationFunction == 'raiseTo5thPower'):
                    temporalRow.append(self.getRaiseToTheFifthPowerDerivative(Fx[row][0]))
                if (activationFunction == 'raiseTo6thPower'):
                    temporalRow.append(self.getRaiseToTheSixthPowerDerivative(Fx[row][0]))
                if (activationFunction == 'exponential'):
                    temporalRow.append(self.getExponentialDerivative(Fx[row][0]))
                dFz.append(temporalRow)
                
            # ----- Get improved and new weigth vector ----- #
            # Djtotal_Dresult = [] # = expected - predicted
            Dresult_Dsum = dFz # = dFz
            # Dsum_Dw = matrix_y # = expected result
            Djtotal_Dw = [] # = (Djtotal_Dresult)*(Dresult_Dsum)*(Dsum_Dw)
            for row in range(0, numberOfIndependentRows):
                temporalRow = []
                current_Djtotal_Dresult = matrix_y[row][0]-Fz[row][0]
                #temporalRow.append(Djtotal_Dresult[row][0]*Dresult_Dsum[row][0]*Dsum_Dw[row][0])
                temporalRow.append(current_Djtotal_Dresult*Dresult_Dsum[row][0])
                Djtotal_Dw.append(temporalRow)
            transposedMatrix_x = matrixMath.getTransposedMatrix(matrix_x)
            learningValue = matrixMath.getDotProduct(transposedMatrix_x, Djtotal_Dw)
            newMatrix_w = []
            for row in range(0, numberOfIndependentVariables+1): # bias + w vector
                temporalRow = []
                temporalRow.append(currentMatrix_w[row][0]+learningRate*learningValue[row][0])
                newMatrix_w.append(temporalRow)
                
            # ----- We save the current weight vector performance ----- #
            Fx = matrixMath.getDotProduct(matrix_x, newMatrix_w)
            Fz = []
            for row in range(0, numberOfIndependentRows):
                temporalRow = []
                if (activationFunction == 'none'):
                    current_Fz = Fx[row][0]
                if (activationFunction == 'sigmoid'):
                    current_Fz = self.getSigmoidActivation(Fx[row][0])
                if (activationFunction == 'relu'):
                    current_Fz = self.getReluActivation(Fx[row][0])
                if (activationFunction == 'tanh'):
                    current_Fz = self.getTanhActivation(Fx[row][0])
                if (activationFunction == 'raiseTo2ndPower'):
                    current_Fz = self.getRaiseToTheSecondPowerActivation(Fx[row][0])
                if (activationFunction == 'raiseTo3rdPower'):
                    current_Fz = self.getRaiseToTheThirdPowerActivation(Fx[row][0])
                if (activationFunction == 'raiseTo4thPower'):
                    current_Fz = self.getRaiseToTheFourthPowerActivation(Fx[row][0])
                if (activationFunction == 'raiseTo5thPower'):
                    current_Fz = self.getRaiseToTheFifthPowerActivation(Fx[row][0])
                if (activationFunction == 'raiseTo6thPower'):
                    current_Fz = self.getRaiseToTheSixthPowerActivation(Fx[row][0])
                if (activationFunction == 'exponential'):
                    current_Fz = self.getExponentialActivation(Fx[row][0]) 
                temporalRow.append(current_Fz)
                Fz.append(temporalRow)
            predictionAcurracy = 0
            predictedData = Fz
            numberOfDataPoints = numberOfIndependentRows
            for row in range(0, numberOfDataPoints):
                n2 = self.y_samplesList[row][0]
                n1 = predictedData[row][0]
                if (isClassification == False):
                    if (((n1*n2) != 0)):
                        newAcurracyValueToAdd = (1-(abs(n2-n1)/abs(n2)))
                        if (newAcurracyValueToAdd < 0):
                            newAcurracyValueToAdd = 0
                        predictionAcurracy = predictionAcurracy + newAcurracyValueToAdd
                if (isClassification == True):
                    if (abs(n1) > abs(n2)): # n2 has to be the one with the highest value with respect to n1
                        n2 = predictedData[row][0]
                        n1 = self.y_samplesList[row][0]
                    if ((n1==0) and (n2>=-1 and n2<=1) and (n2!=0)):
                        predictionAcurracy = predictionAcurracy + ((1-abs(n2))/(1-n1))
                    if (n1==n2):
                        predictionAcurracy = predictionAcurracy + 1
            predictionAcurracy = predictionAcurracy/numberOfDataPoints*100
            temporalRow = []
            temporalRow.append(predictionAcurracy)
            temporalRow.append(newMatrix_w)
            allAccuracies.append(temporalRow)
            # We save the current the modeling results if they are better than
            # the actual best
            currentBestAccuracy = bestModelingResults[1]
            if (predictionAcurracy > currentBestAccuracy):
                bestModelingResults = []
                bestModelingResults.append(newMatrix_w)
                bestModelingResults.append(predictionAcurracy)
                bestModelingResults.append(predictedData)
                bestModelingResults.append(firstMatrix_w)
                bestModelingResults.append("Coefficients distribution is as follows:\nmodelCoefficients =\n[\n  [Neuron1_bias, Neuron1_weight1, Neuron1_weight2, ... , Neuron1_weightM]\n]\n")
            if (predictionAcurracy > stopTrainingIfAcurracy):
                break
        # Alongside the information of the best model obtained, we add the
        # modeled information of ALL the models obtained to the variable that
        # we will return in this method
        bestModelingResults.append(allAccuracies)
        return bestModelingResults
    
    """
    This method is used within the method "getArtificialNeuralNetwork()" to get
    the weights of a particular neuron from a variable that contains all the
    weights of all neurons (matrix_w).
    """
    def getANNweightVectorForOneNeuron(self, matrix_w, neuronNumber):
        temporalRow = []
        for column in range(0, len(matrix_w[neuronNumber])):
            temporalRow.append(matrix_w[neuronNumber][column])
        temporalRow = [temporalRow]
        from ..linearAlgebra import MortrackLinearAlgebraLibrary as mLAL
        matrixMath = mLAL.MatrixMath()
        return matrixMath.getTransposedMatrix(temporalRow)
    
    """
    This method is used within the method "getArtificialNeuralNetwork()" to get
    the partial derivative of the Total Error (dEtotal) due respect with the
    partial derivative of the corresponding Activation Function (dFz) for a
    particular neuron within an Artificial Neural Network.
    """
    def getCurrentDetotal_DFz(self, allMatrix_w, allMatrix_dFz, positionOfNeuronOfCurrentLayer, Detotal_DFzy):
        # Detotal_DFzy[Neuron_n_OfFinalLayer][DerivateOfsample_n][column=0]
        # allMatrix_w[Layer_n][Neuron_n][row=weight_n][column=0]
        # allMatrix_dFz[Layer_n][Neuron_n][row=sample_n_dFzResult][column=0]
        newAllMatrix_w = []
        for currentLayer in range(0, len(allMatrix_w)):
            temporalLayer = []
            for currentNeuronOfCurrentLayer in range(0, len(allMatrix_w[currentLayer])):
                temporalNeuron = []
                if (currentLayer == 0):
                    for currentWeight in range(0, len(allMatrix_w[currentLayer][currentNeuronOfCurrentLayer])):
                        temporalNeuron.append([1])
                else:
                    for currentWeight in range(1, len(allMatrix_w[currentLayer][currentNeuronOfCurrentLayer])):
                        temporalNeuron.append(allMatrix_w[currentLayer][currentNeuronOfCurrentLayer][currentWeight])
                temporalLayer.append(temporalNeuron)
            newAllMatrix_w.append(temporalLayer)
        numberOfSamples = len(allMatrix_dFz[0][0])
        # We create a new matrix that contains all the data of "allMatrix_w"
        # but withouth the biases values and only containing the weight values
        # newAllMatrix_w[Layer_n][Neuron_n][row=weight_n][column=0] # But withouth bias
        numberOfLayers = len(newAllMatrix_w)
        accumulatedWeightCombinations = []
        for current_dFz in range(0, numberOfSamples):
            layerCalculations = [] # [Layer_n][neuron_n][accumulatedMultiplicationOfWeights]
            temporalRow = []
            for cNOCL in range(0, positionOfNeuronOfCurrentLayer):
                temporalRow.append([0])
            temporalRow.append([1])
            layerCalculations.append(temporalRow)
            # We innitialize the variable "layerCalculations" to use in the
            # calculations of the weight combinations
            for currentLayer in range(1, numberOfLayers):
                numberOfNeuronsInCurrentLayer = len(newAllMatrix_w[currentLayer])
                temporalLayer = []
                for currentNeuronOfCurrentLayer in range(0, numberOfNeuronsInCurrentLayer):
                    numberOfNeuronsInPastLayer = len(layerCalculations[len(layerCalculations)-1])
                    temporalRow = []
                    for currentNeuronOfPastLayer in range(0, numberOfNeuronsInPastLayer):
                        for currentPreviousLayerCalculation in range(0, len(layerCalculations[len(layerCalculations)-1][currentNeuronOfPastLayer])):
                            current_aWC = layerCalculations[len(layerCalculations)-1][currentNeuronOfPastLayer][currentPreviousLayerCalculation] * newAllMatrix_w[currentLayer][currentNeuronOfCurrentLayer][currentNeuronOfPastLayer][0] * allMatrix_dFz[currentLayer][currentNeuronOfCurrentLayer][current_dFz][0]
                            if (currentLayer == (numberOfLayers-1)):
                                current_aWC = current_aWC * Detotal_DFzy[currentNeuronOfCurrentLayer][current_dFz][0]
                            temporalRow.append(current_aWC)
                    temporalLayer.append(temporalRow)
                layerCalculations.append(temporalLayer)
            accumulatedWeightCombinations.append(layerCalculations)
        # We now get the values of the acumulated Weight Combinations but for
        # each weight value of the current neuron that we are evaluating
        Detotal_DFz = []
        # accumulatedWeightCombinations[derivateOfCurrentSample][Layer_n][Neuron_n][accumulatedWeightCombinations]
        for current_dFz_Sample in range(0, len(accumulatedWeightCombinations)):
            lastLayer = len(accumulatedWeightCombinations[current_dFz_Sample])-1
            for curentNeuronOfFinalLayer in range(0, len(accumulatedWeightCombinations[current_dFz_Sample][lastLayer])):
                temporalRow = []
                temporalValue = 0
                for currentAccumulatedWeightCombinations in range(0, len(accumulatedWeightCombinations[current_dFz_Sample][lastLayer][0])):
                    temporalValue = temporalValue + accumulatedWeightCombinations[current_dFz_Sample][lastLayer][curentNeuronOfFinalLayer][currentAccumulatedWeightCombinations]
                temporalRow.append(temporalValue)
            Detotal_DFz.append(temporalRow)
        return Detotal_DFz
    
    """
    getArtificialNeuralNetwork(artificialNeuralNetworkDistribution="must contain an array that indicates the distribution of the desired neurons for each layer in columns. If a row-column value equals 1, this will mean that you want a neuron in that position. A 0 means otherwise",
                               activationFunction="the literal name, in lowercaps, of the activation function that you want to apply the neuron. The activation functions must be assigned in an array accordingly to the distribution specified in argument input variable artificialNeuralNetworkDistribution",
                               learningRate="the rate at which you want your Artificial Neural Network to learn (remember that 1=100% learning rate or normal learning rate)",
                               numberOfEpochs="The number of times you want your Artificial Neural Network to train itself",
                               stopTrainingIfAcurracy="define the % value that you want the neuron to stop training itself if such accuracy value is surpassed",
                               isCustomizedInitialWeights="set to True if you will define a customized innitial weight vector for each neuron. False if you want them to be generated randomly",
                               firstMatrix_w="If you set the input argument of this method isCustomizedInitialWeights to True, then assign here the customized innitial weight vectors you desire for each neuron",
                               isClassification="set to True if you are solving a classification problem. False if otherwise")
    
    This method creates an Artificial Neural Network with a customized desired
    number of neurons within it and, within this method, such Artificial Neural
    Network trains itself to learn to predict the input values that it was
    given to study by comparing them with the output expected values.
    When the neuron finishes its learning process, this method will return the
    modeling results.
    
    
    CODE EXAMPLE:
        # matrix_y = [expectedResultForOutputNeuron1, expectedResultForOutputNeuron2]
        matrix_y = [
                [0, 1],
                [1, 0],
                [1, 0],
                [0, 1],
                [1, 0]
                ]
        # matrix_x = [variable1, variable2, variable3]
        matrix_x = [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0]
                ]
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        dL = mSL.DeepLearning(matrix_x, matrix_y)
        # We will indicate that we want 2 neurons in Layer1 and 1 neuron in Layer2
        aNND = [
                [1,1,1],
                [1,1,1]
                ]
        aF = [
              ['relu', 'relu', 'sigmoid'],
              ['relu', 'relu', 'sigmoid']
              ]
        modelingResults = dL.getArtificialNeuralNetwork(artificialNeuralNetworkDistribution=aNND, activationFunction=aF, learningRate=0.1, numberOfEpochs=10000, stopTrainingIfAcurracy=99.9, isCustomizedInitialWeights=False, firstMatrix_w=[], isClassification=True)
        modelCoefficients = modelingResults[0]
        acurracy = modelingResults[1]
        predictedData = modelingResults[2]
        firstMatrix_w = modelingResults[3]
        coefficientDistribution = modelingResults[4]
        allModeledAccuracies = modelingResults[5]
        
    RESULT OF CODE:
        modelCoefficients =
        [
          [2.133298325032156, -0.45548307884431677, -2.1332978269534664, -2.1332978292080043],
          [2.287998188065245, 1.3477978318721369, -1.143999014059006, -1.1439990110690932],
          [-0.6930287605411998, 0.41058709282271444, 0.6057943758418374],
          [4.6826225603458056e-08, -1.8387485390712266, 2.2017181913306803],
          [-4.1791269585765285, -2.5797524896448563, 3.3885776200605955],
          [4.181437529101815, 2.5824655964639742, -3.3907451300458136]
        ]
        
        acurracy =
        98.94028954483407
        
        predictedData =
        [[0.011560111421083964, 0.9884872182827878],
         [0.9873319964204451, 0.01262867979045398],
         [0.9873319961998808, 0.012628680010459043],
         [0.015081447917016324, 0.9849528347708301],
         [0.9989106156594524, 0.0010867877109744279]]
        
        coefficientDistribution =
        "
        Coefficients distribution is as follows:
        modelCoefficients =
        [
          [Neuron1_bias, Neuron1_weight1, Neuron1_weight2, ... , Neuron1_weightM],
          [Neuron2_bias, Neuron2_weight1, Neuron2_weight2, ... , Neuron2_weightZ],
          [     .      ,        .       ,        .       , ... ,        .       ],
          [     .      ,        .       ,        .       , ... ,        .       ],
          [     .      ,        .       ,        .       , ... ,        .       ],
          [NeuronN_bias, NeuronN_weight1, NeuronN_weight2, ... , NeuronN_weightK],
        ]
        "
        
        allModeledAccuracies["independent variable distribution used to get a model"]["model accuracy", "model coefficients obtained but with original distribution"] =
        # NOTE: since this variable contains large amounts of information, it
        #       will not be displayed but only described on how to use it.
    """
    def getArtificialNeuralNetwork(self, artificialNeuralNetworkDistribution, activationFunction, learningRate=1, numberOfEpochs=1000, stopTrainingIfAcurracy=95, isCustomizedInitialWeights=False, firstMatrix_w=[], isClassification=True):
        from ..linearAlgebra import MortrackLinearAlgebraLibrary as mLAL
        from . import MortrackML_Library as mSL
        import random
        numberOfIndependentRows= len(self.x_samplesList)
        numberOfIndependentVariables = len(self.x_samplesList[0])
        numberOfNeuronLayers = len(artificialNeuralNetworkDistribution[0])
        numberOfNeuronsPerLayer = []
        activationFunctionsList = []
        totalNumberOfNeurons = 0
        matrixMath = mLAL.MatrixMath()
        transposedANND = matrixMath.getTransposedMatrix(artificialNeuralNetworkDistribution)
        transposedAF = matrixMath.getTransposedMatrix(activationFunction)
        for row in range(0, len(transposedANND)):
            currentNumberOfNeurons = 0
            for column in range(0, len(transposedANND[0])):
                if (transposedANND[row][column] == 1):
                    currentNumberOfNeurons = currentNumberOfNeurons + 1
                    activationFunctionsList.append(transposedAF[row][column])
            temporalRow = []
            temporalRow.append(currentNumberOfNeurons)
            numberOfNeuronsPerLayer.append(temporalRow)
            totalNumberOfNeurons = totalNumberOfNeurons + currentNumberOfNeurons
        numberOfNeuronsPerLayer = matrixMath.getTransposedMatrix(numberOfNeuronsPerLayer)
        activationFunctionsList = [activationFunctionsList]
        numberOfNeuronsInFinalLayer = numberOfNeuronsPerLayer[0][len(numberOfNeuronsPerLayer[0])-1]
        for column in range(0, numberOfNeuronLayers):
            for row in range(0, numberOfNeuronsPerLayer[0][column]):
                if ((activationFunction[row][column]!='none') and (activationFunction[row][column]!='sigmoid') and (activationFunction[row][column]!='relu') and (activationFunction[row][column]!='tanh') and (activationFunction[row][column]!='raiseTo2ndPower') and (activationFunction[row][column]!='raiseTo3rdPower') and (activationFunction[row][column]!='raiseTo4thPower') and (activationFunction[row][column]!='raiseTo5thPower') and (activationFunction[row][column]!='raiseTo6thPower') and (activationFunction[row][column]!='exponential')):
                    raise Exception('ERROR: The selected Activation Function does not exist or has not been programmed in this method yet.')
        
        totalNumberOfLayers = len(numberOfNeuronsPerLayer[0])
        matrix_x = []
        for row in range(0, numberOfIndependentRows):
            temporalRow = []
            temporalRow.append(1)
            for column in range(0, numberOfIndependentVariables):
                temporalRow.append(self.x_samplesList[row][column])
            matrix_x.append(temporalRow)
        matrix_y = self.y_samplesList
        # We innitialize the weight vector random values from -1 up to +1
        if (isCustomizedInitialWeights == False):
            matrix_w = []
            for currentLayer in range(0, totalNumberOfLayers):
                for column in range(0, numberOfNeuronsPerLayer[0][currentLayer]):
                    temporalRow = []
                    if (currentLayer == 0):
                        for row in range(0, numberOfIndependentVariables+1):
                            temporalRow.append(random.random()*2-1)
                    else:
                        for row in range(0, numberOfNeuronsPerLayer[0][currentLayer-1]+1):
                            temporalRow.append(random.random()*2-1)
                    matrix_w.append(temporalRow)
            firstMatrix_w = matrix_w
        else:
            matrix_w = firstMatrix_w
        
        # We calculate the results obtained with the innitialized random weight
        # vector (We calculate the matrixes for Fx, Fz and dFz)
        Fx = []
        Fz = []
        dFz = []
        actualFunctionActivation = 0
        for currentLayer in range(0, totalNumberOfLayers):
            temporalRow1 = []
            if (currentLayer == 0):
                for currentNeuronOfCurrentLayer in range(0, numberOfNeuronsPerLayer[0][currentLayer]):
                    temporalRow1 = matrixMath.getDotProduct(matrix_x, self.getANNweightVectorForOneNeuron(matrix_w, currentNeuronOfCurrentLayer))
                    temporalRow1 = matrixMath.getTransposedMatrix(temporalRow1)
                    Fx.append(temporalRow1)
                for currentNeuronOfCurrentLayer in range(0, numberOfNeuronsPerLayer[0][currentLayer]):
                    temporalRow1 = []
                    temporalRow2= []
                    for column in range(0, numberOfIndependentRows):
                        # Activation Functions (Fz)
                        if (activationFunctionsList[0][actualFunctionActivation] == 'none'):
                            current_Fz = Fx[currentNeuronOfCurrentLayer][0][column]
                        if (activationFunctionsList[0][actualFunctionActivation] == 'sigmoid'):
                            current_Fz = self.getSigmoidActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'relu'):
                            current_Fz = self.getReluActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'tanh'):
                            current_Fz = self.getTanhActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo2ndPower'):
                            current_Fz = self.getRaiseToTheSecondPowerActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo3rdPower'):
                            current_Fz = self.getRaiseToTheThirdPowerActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo4thPower'):
                            current_Fz = self.getRaiseToTheFourthPowerActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo5thPower'):
                            current_Fz = self.getRaiseToTheFifthPowerActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo6thPower'):
                            current_Fz = self.getRaiseToTheSixthPowerActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'exponential'):
                            current_Fz = self.getExponentialActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                        # Derivates (dFz)
                        if (activationFunctionsList[0][actualFunctionActivation] == 'none'):
                            if (current_Fz != 0):
                                current_dFz = 1
                            else:
                                current_dFz = 0
                        if (activationFunctionsList[0][actualFunctionActivation] == 'sigmoid'):
                            current_dFz = current_Fz*(1-current_Fz)
                        if (activationFunctionsList[0][actualFunctionActivation] == 'relu'):
                            current_dFz = self.getReluActivationDerivative(Fx[currentNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'tanh'):
                            current_dFz = 1-current_Fz**2
                        if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo2ndPower'):
                            current_dFz = self.getRaiseToTheSecondPowerDerivative(Fx[currentNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo3rdPower'):
                            current_dFz = self.getRaiseToTheThirdPowerDerivative(Fx[currentNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo4thPower'):
                            current_dFz = self.getRaiseToTheFourthPowerDerivative(Fx[currentNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo5thPower'):
                            current_dFz = self.getRaiseToTheFifthPowerDerivative(Fx[currentNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo6thPower'):
                            current_dFz = self.getRaiseToTheSixthPowerDerivative(Fx[currentNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'exponential'):
                            current_dFz = self.getExponentialDerivative(Fx[currentNeuronOfCurrentLayer][0][column])
                        temporalRow1.append(current_Fz)
                        temporalRow2.append(current_dFz)
                    actualFunctionActivation = actualFunctionActivation + 1
                    Fz.append(temporalRow1)
                    dFz.append(temporalRow2)
            else:
                pastNeuronOfCurrentLayer = 0
                for currentLayerCount in range(0, currentLayer-1):
                    pastNeuronOfCurrentLayer = pastNeuronOfCurrentLayer + numberOfNeuronsPerLayer[0][currentLayerCount]
                inputMatrix = []
                for row in range(0, numberOfIndependentRows):
                    temporalRow1 = []
                    temporalRow1.append(1) # bias column
                    for currentNeuron in range(pastNeuronOfCurrentLayer, pastNeuronOfCurrentLayer+numberOfNeuronsPerLayer[0][currentLayer-1]):
                        temporalRow1.append(Fz[currentNeuron][row])
                    inputMatrix.append(temporalRow1)
                for currentNeuronOfCurrentLayer in range(pastNeuronOfCurrentLayer+numberOfNeuronsPerLayer[0][currentLayer-1], pastNeuronOfCurrentLayer+numberOfNeuronsPerLayer[0][currentLayer-1]+numberOfNeuronsPerLayer[0][currentLayer]):
                    temporalRow1 = matrixMath.getDotProduct(inputMatrix, self.getANNweightVectorForOneNeuron(matrix_w, currentNeuronOfCurrentLayer))
                    temporalRow1 = matrixMath.getTransposedMatrix(temporalRow1)
                    Fx.append(temporalRow1)
                pastNeuronOfCurrentLayer = 0
                for currentLayerCount in range(0, currentLayer):
                    pastNeuronOfCurrentLayer = pastNeuronOfCurrentLayer + numberOfNeuronsPerLayer[0][currentLayerCount]
                for currentNeuronOfCurrentLayer in range(0, numberOfNeuronsPerLayer[0][currentLayer]):
                    temporalRow1= []
                    temporalRow2= []
                    for column in range(0, numberOfIndependentRows):
                        # Activation Functions (Fz)
                        if (activationFunctionsList[0][actualFunctionActivation] == 'none'):
                            current_Fz = Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column]
                        if (activationFunctionsList[0][actualFunctionActivation] == 'sigmoid'):
                            current_Fz = self.getSigmoidActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'relu'):
                            current_Fz = self.getReluActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'tanh'):
                            current_Fz = self.getTanhActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo2ndPower'):
                            current_Fz = self.getRaiseToTheSecondPowerActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo3rdPower'):
                            current_Fz = self.getRaiseToTheThirdPowerActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo4thPower'):
                            current_Fz = self.getRaiseToTheFourthPowerActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo5thPower'):
                            current_Fz = self.getRaiseToTheFifthPowerActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo6thPower'):
                            current_Fz = self.getRaiseToTheSixthPowerActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'exponential'):
                            current_Fz = self.getExponentialActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                        # Derivates (dFz)
                        if (activationFunctionsList[0][actualFunctionActivation] == 'none'):
                            if (current_Fz != 0):
                                current_dFz = 1
                            else:
                                current_dFz = 0
                        if (activationFunctionsList[0][actualFunctionActivation] == 'sigmoid'):
                            current_dFz = current_Fz*(1-current_Fz)
                        if (activationFunctionsList[0][actualFunctionActivation] == 'relu'):
                            current_dFz = self.getReluActivationDerivative(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'tanh'):
                            current_dFz = 1-current_Fz**2
                        if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo2ndPower'):
                            current_dFz = self.getRaiseToTheSecondPowerDerivative(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo3rdPower'):
                            current_dFz = self.getRaiseToTheThirdPowerDerivative(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo4thPower'):
                            current_dFz = self.getRaiseToTheFourthPowerDerivative(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo5thPower'):
                            current_dFz = self.getRaiseToTheFifthPowerDerivative(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo6thPower'):
                            current_dFz = self.getRaiseToTheSixthPowerDerivative(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'exponential'):
                            current_dFz = self.getExponentialDerivative(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                        temporalRow1.append(current_Fz)
                        temporalRow2.append(current_dFz)
                    actualFunctionActivation = actualFunctionActivation + 1
                    Fz.append(temporalRow1)
                    dFz.append(temporalRow2)
                    
        # We evaluate the performance of the innitialized weight vectors
        predictionAcurracy = 0
        predictedData = []
        for currentNeuronOfLastLayer in range(0, numberOfNeuronsInFinalLayer):
            predictedData.append(Fz[totalNumberOfNeurons-numberOfNeuronsInFinalLayer+currentNeuronOfLastLayer])
        predictedData = matrixMath.getTransposedMatrix(predictedData)
        numberOfDataPoints = numberOfIndependentRows*numberOfNeuronsInFinalLayer
        for currentNeuronOfLastLayer in range(0, numberOfNeuronsInFinalLayer):
            for row in range(0, numberOfIndependentRows):
                n2 = self.y_samplesList[row][currentNeuronOfLastLayer]
                n1 = predictedData[row][currentNeuronOfLastLayer]
                if (isClassification == False):
                    if (((n1*n2) != 0)):
                    #if (((n1*n2) > 0) and (n1!=n2)):
                        newAcurracyValueToAdd = (1-(abs(n2-n1)/abs(n2)))
                        if (newAcurracyValueToAdd < 0):
                                newAcurracyValueToAdd = 0
                        predictionAcurracy = predictionAcurracy + newAcurracyValueToAdd
                        #predictionAcurracy = predictionAcurracy + (n1/n2)
                if (isClassification == True):
                    if (abs(n1) > abs(n2)): # n2 has to be the one with the highest value with respect to n1
                        n2 = predictedData[row][currentNeuronOfLastLayer]
                        n1 = self.y_samplesList[row][currentNeuronOfLastLayer]
                    if ((n1==0) and (n2>=-1 and n2<=1) and (n2!=0)):
                        predictionAcurracy = predictionAcurracy + ((1-abs(n2))/(1-n1))
                    if (n1==n2):
                    #if ((n1==n2) and (n1==0)):
                        predictionAcurracy = predictionAcurracy + 1
        predictionAcurracy = predictionAcurracy/numberOfDataPoints*100
        
        # We save the current the modeling results
        bestModelingResults = []
        bestModelingResults.append(matrix_w)
        bestModelingResults.append(predictionAcurracy)
        bestModelingResults.append(predictedData)
        bestModelingResults.append(firstMatrix_w)
        bestModelingResults.append("Coefficients distribution is as follows:\nmodelCoefficients =\n[\n  [Neuron1_bias, Neuron1_weight1, Neuron1_weight2, ... , Neuron1_weightM],\n  [Neuron2_bias, Neuron2_weight1, Neuron2_weight2, ... , Neuron2_weightZ],\n  [     .      ,        .       ,        .       , ... ,        .       ],\n  [     .      ,        .       ,        .       , ... ,        .       ],\n  [     .      ,        .       ,        .       , ... ,        .       ],\n  [NeuronN_bias, NeuronN_weight1, NeuronN_weight2, ... , NeuronN_weightK],\n]\n")
        allAccuracies = []
        temporalRow = []
        temporalRow.append(bestModelingResults[1])
        temporalRow.append(bestModelingResults[0])
        allAccuracies.append(temporalRow)
        
        
        # ---------------------------------------------------------------- #
        # ----- WE START THE TRAINING PROCESS FROM EPOCH 2 AND ABOVE ----- #
        # ---------------------------------------------------------------- #
        # 2nd Epoch
        temporalMatrixOfMatrix_w = []
        Detotal_DFzy = [] # Detotal_DFzy[Neuron_n][sample_n][column=0]
        for currentLayer in range(0, totalNumberOfLayers):
            trueCurrentLayer = totalNumberOfLayers-currentLayer
            pastNeuronsOfCurrentLayer = 0
            for currentLayerCount in range(0, (trueCurrentLayer-1)):
                pastNeuronsOfCurrentLayer = pastNeuronsOfCurrentLayer + numberOfNeuronsPerLayer[0][currentLayerCount]
            for currentNeuronOfCurrentLayer in range(0, numberOfNeuronsPerLayer[0][trueCurrentLayer-1]):
                trueCurrentNeuronOfCurrentLayer = numberOfNeuronsPerLayer[0][trueCurrentLayer-1]-1-currentNeuronOfCurrentLayer
                if (currentLayer == 0):
                    # ----- We first update the weights of the output neuron ----- #
                    Detotal_Dfz = [] # = predicted - expected
                    expectedOutput = matrix_y
                    predictedOutput = []
                    Dfz_Df = [] # = dFz
                    predictedOutput.append(Fz[pastNeuronsOfCurrentLayer+trueCurrentNeuronOfCurrentLayer])
                    Dfz_Df.append(dFz[pastNeuronsOfCurrentLayer+trueCurrentNeuronOfCurrentLayer])
                    predictedOutput = matrixMath.getTransposedMatrix(predictedOutput)
                    Dfz_Df = matrixMath.getTransposedMatrix(Dfz_Df)
                    Df_Dw = []
                    Detotal_Dw = [] # = (Detotal_Dfz)*(Dfz_Df)*(Df_Dw)
                    # We calculate "Detotal_Dfz"
                    for row in range(0, numberOfIndependentRows):
                       temporalRow = []
                       temporalRow.append(predictedOutput[row][0] - expectedOutput[row][trueCurrentNeuronOfCurrentLayer])
                       Detotal_Dfz.append(temporalRow)
                    # We calculate "Df_Dw"
                    if (totalNumberOfLayers == 1):
                        Df_Dw = matrixMath.getTransposedMatrix(matrix_x)
                    else:
                        temporalNeuronsAnalized = 0
                        for n in range(0, trueCurrentLayer-2):
                            temporalNeuronsAnalized = temporalNeuronsAnalized + numberOfNeuronsPerLayer[0][n]
                        temporalRow = []
                        for currentBiasDerivate in range(0, numberOfIndependentRows):
                            temporalRow.append(1)
                        Df_Dw.append(temporalRow)
                        for currentNeuronOfPastLayer in range(0, numberOfNeuronsPerLayer[0][trueCurrentLayer-2]):
                            Df_Dw.append(dFz[temporalNeuronsAnalized+currentNeuronOfPastLayer])
                    # We calculate "Detotal_Dw"
                    Detotal_Dfz_TIMES_Dfz_Df = []
                    for currentSample in range(0, len(Detotal_Dfz)):
                        Detotal_Dfz_TIMES_Dfz_Df.append([Detotal_Dfz[currentSample][0] * Dfz_Df[currentSample][0]])
                    Detotal_Dw = matrixMath.getDotProduct(Df_Dw, Detotal_Dfz_TIMES_Dfz_Df)
                    # We finnally update the weight values of the last neuron
                    temporalNeuronsAnalized = 0
                    for n in range(0, trueCurrentLayer-1):
                        temporalNeuronsAnalized = temporalNeuronsAnalized + numberOfNeuronsPerLayer[0][n]
                    currentVector_w = matrix_w[temporalNeuronsAnalized+trueCurrentNeuronOfCurrentLayer]
                    currentVector_w = matrixMath.getTransposedMatrix([currentVector_w])
                    temporalRow = []
                    for currentWeight in range(0, len(matrix_w[temporalNeuronsAnalized+trueCurrentNeuronOfCurrentLayer])):
                        temporalRow.append(currentVector_w[currentWeight][0]-learningRate*Detotal_Dw[currentWeight][0])
                    temporalMatrixOfMatrix_w.append(temporalRow)
                    Detotal_DFzy.append(Detotal_Dfz_TIMES_Dfz_Df)
                else:
                    fixed_Detotal_DFzy = []
                    for row in range(0, numberOfNeuronsInFinalLayer):
                        fixed_Detotal_DFzy.append(Detotal_DFzy[numberOfNeuronsInFinalLayer-row-1])
                    # ----- We Now update the weights of the other neurons ----- #
                    Detotal_Dfz = [] # = predicted - expected
                    Df_Dw = []
                    Detotal_Dw = [] # = (Detotal_Dfz)*(Dfz_Df)*(Df_Dw)
                    # We calculate "Detotal_Dfz"
                    # trueCurrentLayer
                    Detotal_Dfz = Detotal_DFzy 
                    
                    # We create a temporal matrix for "matrix_w" and "matrix_dFz"
                    # to just re-arrange the structure of how both matrixes
                    # have their actual data. This is needed to then use the
                    # method "self.getCurrentDetotal_DFz()" and to get
                    # Detotal_Dfz through such method.
                    temporalMatrix_w = []
                    temporalDfzMatrix = []
                    
                    # Este loop se repite N veces = "penultima layer" - "1 layer adelante de la actual"
                    for currentFurtherLayer in range(trueCurrentLayer, totalNumberOfLayers):
                        # "temporalNeuronsAnalized" tiene el numero de neuronas que hay en todas las layers anteriores a la actual
                        if (len(temporalMatrix_w) == 0):
                            temporalRow = []
                            temporalNeuronsAnalized = 0
                            for n in range(0, trueCurrentLayer-1):
                                temporalNeuronsAnalized = temporalNeuronsAnalized + numberOfNeuronsPerLayer[0][n]
                            # We get "Dfz_Df" of the neuron that will improve its weights
                            Dfz_Df = dFz[temporalNeuronsAnalized+trueCurrentNeuronOfCurrentLayer] # = dFz
                            
                            # We get the weights of the neuron that will improve its weights
                            currentVector_w = matrix_w[temporalNeuronsAnalized+trueCurrentNeuronOfCurrentLayer]
                            currentVector_w = matrixMath.getTransposedMatrix([currentVector_w])
                            # We plug in all the weight vectors
                            temporalRow.append(currentVector_w)
                            temporalMatrix_w.append(temporalRow)
                            # We plug in the dFz of the last neuron
                            temporalRow = []
                            currentVector_w = dFz[temporalNeuronsAnalized+trueCurrentNeuronOfCurrentLayer]
                            currentVector_w = matrixMath.getTransposedMatrix([currentVector_w])
                            # We plug in all the weight vectors
                            temporalRow.append(currentVector_w)
                            temporalDfzMatrix.append(temporalRow)
                            
                            # dFz de donde viene la actual weight o del independent variable en caso de tratarse de la 1ra layer
                            # We calculate "Df_Dw"
                            if (trueCurrentLayer == 1):
                                Df_Dw = matrixMath.getTransposedMatrix(matrix_x)
                            else:
                                temporalNeuronsAnalized = 0
                                for n in range(0, trueCurrentLayer-2):
                                    temporalNeuronsAnalized = temporalNeuronsAnalized + numberOfNeuronsPerLayer[0][n]
                                temporalRow = []
                                for currentBiasDerivate in range(0, numberOfIndependentRows):
                                    temporalRow.append(1) # bias derivate
                                Df_Dw.append(temporalRow)
                                for currentPastNeuronOfPastLayer in range(0, numberOfNeuronsPerLayer[0][trueCurrentLayer-2]):
                                    Df_Dw.append(dFz[temporalNeuronsAnalized+currentPastNeuronOfPastLayer])
                        
                        temporalRow = []
                        temporalNeuronsAnalized = 0
                        for n in range(0, currentFurtherLayer):
                            temporalNeuronsAnalized = temporalNeuronsAnalized + numberOfNeuronsPerLayer[0][n]
                        # Este loop se repite N veces = numero de neuronas en la actual layer (la cual empieza a partir de la layer futura / posterior)
                        for currentFutherNeuronOfCurrentLayer in range(0, numberOfNeuronsPerLayer[0][currentFurtherLayer]):
                            currentVector_w = matrix_w[temporalNeuronsAnalized+currentFutherNeuronOfCurrentLayer]
                            currentVector_w = matrixMath.getTransposedMatrix([currentVector_w])
                            # We plug in all the weight vectors
                            temporalRow.append(currentVector_w)
                        temporalMatrix_w.append(temporalRow)
                        temporalRow = []
                        # Este loop se repite N veces = numero de neuronas en la actual layer (la cual empieza a partir de la layer futura / posterior)
                        for currentFutherNeuronOfCurrentLayer in range(0, numberOfNeuronsPerLayer[0][currentFurtherLayer]):
                            currentVector_w = dFz[temporalNeuronsAnalized+currentFutherNeuronOfCurrentLayer]
                            currentVector_w = matrixMath.getTransposedMatrix([currentVector_w])
                            # We plug in all the weight vectors
                            temporalRow.append(currentVector_w)
                        temporalDfzMatrix.append(temporalRow)
                        
                    # Detotal_DFzy[DerivateOfsample_n][column=0]
                    Detotal_Dfz = self.getCurrentDetotal_DFz(temporalMatrix_w, temporalDfzMatrix, trueCurrentNeuronOfCurrentLayer, fixed_Detotal_DFzy)
                    
                    # We calculate "Detotal_Dw"
                    Detotal_Dfz_TIMES_Dfz_Df = []
                    for currentSample in range(0, len(Detotal_Dfz)):
                        Detotal_Dfz_TIMES_Dfz_Df.append([Detotal_Dfz[currentSample][0] * Dfz_Df[currentSample]])
                    Detotal_Dw = matrixMath.getDotProduct(Df_Dw, Detotal_Dfz_TIMES_Dfz_Df)
                    
                    # We finnally update the weight values of the last neuron
                    # temporalMatrixOfMatrix_w = []
                    temporalNeuronsAnalized = 0
                    for n in range(0, trueCurrentLayer-1):
                        temporalNeuronsAnalized = temporalNeuronsAnalized + numberOfNeuronsPerLayer[0][n]
                    # We get the weights of the neuron that will improve its weights
                    currentVector_w = matrix_w[temporalNeuronsAnalized+trueCurrentNeuronOfCurrentLayer]
                    currentVector_w = matrixMath.getTransposedMatrix([currentVector_w])
                    temporalRow = []
                    for currentWeight in range(0, len(matrix_w[temporalNeuronsAnalized+trueCurrentNeuronOfCurrentLayer])):
                        temporalRow.append(currentVector_w[currentWeight][0]-learningRate*Detotal_Dw[currentWeight][0])
                    temporalMatrixOfMatrix_w.append(temporalRow)
                          
        # We reorder the new obtained weights but accordingly to the neurons
        # order (from neuron 1 to neuron "N") in variable "newMatrix_w"
        newMatrix_w = []
        for row in range(0, totalNumberOfNeurons):
            newMatrix_w.append(temporalMatrixOfMatrix_w[totalNumberOfNeurons-row-1])
        
        # 3rd Epoch and above
        for currentEpoch in range(1, numberOfEpochs):
            print('Current Epoch = ' + format(currentEpoch))
            # ----- Predict the output values with latest weight vector ----- #
            currentMatrix_w = newMatrix_w
            Fx = []
            Fz = []
            dFz = []
            actualFunctionActivation = 0
            for currentLayer in range(0, totalNumberOfLayers):
                temporalRow1 = []
                if (currentLayer == 0):
                    for currentNeuronOfCurrentLayer in range(0, numberOfNeuronsPerLayer[0][currentLayer]):
                        temporalRow1 = matrixMath.getDotProduct(matrix_x, self.getANNweightVectorForOneNeuron(currentMatrix_w, currentNeuronOfCurrentLayer))
                        temporalRow1 = matrixMath.getTransposedMatrix(temporalRow1)
                        Fx.append(temporalRow1)
                    for currentNeuronOfCurrentLayer in range(0, numberOfNeuronsPerLayer[0][currentLayer]):
                        temporalRow1 = []
                        temporalRow2= []
                        for column in range(0, numberOfIndependentRows):
                            # Activation Functions (Fz)
                            if (activationFunctionsList[0][actualFunctionActivation] == 'none'):
                                current_Fz = Fx[currentNeuronOfCurrentLayer][0][column]
                            if (activationFunctionsList[0][actualFunctionActivation] == 'sigmoid'):
                                current_Fz = self.getSigmoidActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'relu'):
                                current_Fz = self.getReluActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'tanh'):
                                current_Fz = self.getTanhActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo2ndPower'):
                                current_Fz = self.getRaiseToTheSecondPowerActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo3rdPower'):
                                current_Fz = self.getRaiseToTheThirdPowerActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo4thPower'):
                                current_Fz = self.getRaiseToTheFourthPowerActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo5thPower'):
                                current_Fz = self.getRaiseToTheFifthPowerActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo6thPower'):
                                current_Fz = self.getRaiseToTheSixthPowerActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'exponential'):
                                current_Fz = self.getExponentialActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                            # Derivates (dFz)
                            if (activationFunctionsList[0][actualFunctionActivation] == 'none'):
                                if (current_Fz != 0):
                                    current_dFz = 1
                                else:
                                    current_dFz = 0
                            if (activationFunctionsList[0][actualFunctionActivation] == 'sigmoid'):
                                current_dFz = current_Fz*(1-current_Fz)
                            if (activationFunctionsList[0][actualFunctionActivation] == 'relu'):
                                current_dFz = self.getReluActivationDerivative(Fx[currentNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'tanh'):
                                current_dFz = 1-current_Fz**2
                            if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo2ndPower'):
                                current_dFz = self.getRaiseToTheSecondPowerDerivative(Fx[currentNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo3rdPower'):
                                current_dFz = self.getRaiseToTheThirdPowerDerivative(Fx[currentNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo4thPower'):
                                current_dFz = self.getRaiseToTheFourthPowerDerivative(Fx[currentNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo5thPower'):
                                current_dFz = self.getRaiseToTheFifthPowerDerivative(Fx[currentNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo6thPower'):
                                current_dFz = self.getRaiseToTheSixthPowerDerivative(Fx[currentNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'exponential'):
                                current_dFz = self.getExponentialDerivative(Fx[currentNeuronOfCurrentLayer][0][column])
                            temporalRow1.append(current_Fz)
                            temporalRow2.append(current_dFz)
                        actualFunctionActivation = actualFunctionActivation + 1
                        Fz.append(temporalRow1)
                        dFz.append(temporalRow2)
                else:
                    pastNeuronOfCurrentLayer = 0
                    for currentLayerCount in range(0, currentLayer-1):
                        pastNeuronOfCurrentLayer = pastNeuronOfCurrentLayer + numberOfNeuronsPerLayer[0][currentLayerCount]
                    inputMatrix = []
                    for row in range(0, numberOfIndependentRows):
                        temporalRow1 = []
                        temporalRow1.append(1) # bias column
                        for currentNeuron in range(pastNeuronOfCurrentLayer, pastNeuronOfCurrentLayer+numberOfNeuronsPerLayer[0][currentLayer-1]):
                            temporalRow1.append(Fz[currentNeuron][row])
                        inputMatrix.append(temporalRow1)
                    for currentNeuronOfCurrentLayer in range(pastNeuronOfCurrentLayer+numberOfNeuronsPerLayer[0][currentLayer-1], pastNeuronOfCurrentLayer+numberOfNeuronsPerLayer[0][currentLayer-1]+numberOfNeuronsPerLayer[0][currentLayer]):
                        temporalRow1 = matrixMath.getDotProduct(inputMatrix, self.getANNweightVectorForOneNeuron(currentMatrix_w, currentNeuronOfCurrentLayer))
                        temporalRow1 = matrixMath.getTransposedMatrix(temporalRow1)
                        Fx.append(temporalRow1)
                    pastNeuronOfCurrentLayer = 0
                    for currentLayerCount in range(0, currentLayer):
                        pastNeuronOfCurrentLayer = pastNeuronOfCurrentLayer + numberOfNeuronsPerLayer[0][currentLayerCount]
                    for currentNeuronOfCurrentLayer in range(0, numberOfNeuronsPerLayer[0][currentLayer]):
                        temporalRow1= []
                        temporalRow2= []
                        for column in range(0, numberOfIndependentRows):
                            # Activation Functions (Fz)
                            if (activationFunctionsList[0][actualFunctionActivation] == 'none'):
                                current_Fz = Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column]
                            if (activationFunctionsList[0][actualFunctionActivation] == 'sigmoid'):
                                current_Fz = self.getSigmoidActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'relu'):
                                current_Fz = self.getReluActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'tanh'):
                                current_Fz = self.getTanhActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo2ndPower'):
                                current_Fz = self.getRaiseToTheSecondPowerActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo3rdPower'):
                                current_Fz = self.getRaiseToTheThirdPowerActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo4thPower'):
                                current_Fz = self.getRaiseToTheFourthPowerActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo5thPower'):
                                current_Fz = self.getRaiseToTheFifthPowerActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo6thPower'):
                                current_Fz = self.getRaiseToTheSixthPowerActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'exponential'):
                                current_Fz = self.getExponentialActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                            # Derivates (dFz)
                            if (activationFunctionsList[0][actualFunctionActivation] == 'none'):
                                if (current_Fz != 0):
                                    current_dFz = 1
                                else:
                                    current_dFz = 0
                            if (activationFunctionsList[0][actualFunctionActivation] == 'sigmoid'):
                                current_dFz = current_Fz*(1-current_Fz)
                            if (activationFunctionsList[0][actualFunctionActivation] == 'relu'):
                                current_dFz = self.getReluActivationDerivative(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'tanh'):
                                current_dFz = 1-current_Fz**2
                            if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo2ndPower'):
                                current_dFz = self.getRaiseToTheSecondPowerDerivative(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo3rdPower'):
                                current_dFz = self.getRaiseToTheThirdPowerDerivative(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo4thPower'):
                                current_dFz = self.getRaiseToTheFourthPowerDerivative(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo5thPower'):
                                current_dFz = self.getRaiseToTheFifthPowerDerivative(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo6thPower'):
                                current_dFz = self.getRaiseToTheSixthPowerDerivative(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'exponential'):
                                current_dFz = self.getExponentialDerivative(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                            temporalRow1.append(current_Fz)
                            temporalRow2.append(current_dFz)
                        actualFunctionActivation = actualFunctionActivation + 1
                        Fz.append(temporalRow1)
                        dFz.append(temporalRow2)
            
            # ----- Get improved and new weigth vector ----- #
            temporalMatrixOfMatrix_w = []
            Detotal_DFzy = [] # Detotal_DFzy[Neuron_n][sample_n][column=0]
            for currentLayer in range(0, totalNumberOfLayers):
                trueCurrentLayer = totalNumberOfLayers-currentLayer
                pastNeuronsOfCurrentLayer = 0
                for currentLayerCount in range(0, (trueCurrentLayer-1)):
                    pastNeuronsOfCurrentLayer = pastNeuronsOfCurrentLayer + numberOfNeuronsPerLayer[0][currentLayerCount]
                for currentNeuronOfCurrentLayer in range(0, numberOfNeuronsPerLayer[0][trueCurrentLayer-1]):
                    trueCurrentNeuronOfCurrentLayer = numberOfNeuronsPerLayer[0][trueCurrentLayer-1]-1-currentNeuronOfCurrentLayer
                    if (currentLayer == 0):
                        # ----- We first update the weights of the output neuron ----- #
                        Detotal_Dfz = [] # = predicted - expected
                        expectedOutput = matrix_y
                        predictedOutput = []
                        Dfz_Df = [] # = dFz
                        predictedOutput.append(Fz[pastNeuronsOfCurrentLayer+trueCurrentNeuronOfCurrentLayer])
                        Dfz_Df.append(dFz[pastNeuronsOfCurrentLayer+trueCurrentNeuronOfCurrentLayer])
                        predictedOutput = matrixMath.getTransposedMatrix(predictedOutput)
                        Dfz_Df = matrixMath.getTransposedMatrix(Dfz_Df)
                        Df_Dw = []
                        Detotal_Dw = [] # = (Detotal_Dfz)*(Dfz_Df)*(Df_Dw)
                        # We calculate "Detotal_Dfz"
                        for row in range(0, numberOfIndependentRows):
                           temporalRow = []
                           temporalRow.append(predictedOutput[row][0] - expectedOutput[row][trueCurrentNeuronOfCurrentLayer])
                           Detotal_Dfz.append(temporalRow)
                        # We calculate "Df_Dw"
                        if (totalNumberOfLayers == 1):
                            Df_Dw = matrixMath.getTransposedMatrix(matrix_x)
                        else:
                            temporalNeuronsAnalized = 0
                            for n in range(0, trueCurrentLayer-2):
                                temporalNeuronsAnalized = temporalNeuronsAnalized + numberOfNeuronsPerLayer[0][n]
                            temporalRow = []
                            for currentBiasDerivate in range(0, numberOfIndependentRows):
                                temporalRow.append(1)
                            Df_Dw.append(temporalRow)
                            for currentNeuronOfPastLayer in range(0, numberOfNeuronsPerLayer[0][trueCurrentLayer-2]):
                                Df_Dw.append(dFz[temporalNeuronsAnalized+currentNeuronOfPastLayer])
                        # We calculate "Detotal_Dw"
                        Detotal_Dfz_TIMES_Dfz_Df = []
                        for currentSample in range(0, len(Detotal_Dfz)):
                            Detotal_Dfz_TIMES_Dfz_Df.append([Detotal_Dfz[currentSample][0] * Dfz_Df[currentSample][0]])
                        Detotal_Dw = matrixMath.getDotProduct(Df_Dw, Detotal_Dfz_TIMES_Dfz_Df)
                        # We finnally update the weight values of the last neuron
                        temporalNeuronsAnalized = 0
                        for n in range(0, trueCurrentLayer-1):
                            temporalNeuronsAnalized = temporalNeuronsAnalized + numberOfNeuronsPerLayer[0][n]
                        currentVector_w = currentMatrix_w[temporalNeuronsAnalized+trueCurrentNeuronOfCurrentLayer]
                        currentVector_w = matrixMath.getTransposedMatrix([currentVector_w])
                        temporalRow = []
                        for currentWeight in range(0, len(currentMatrix_w[temporalNeuronsAnalized+trueCurrentNeuronOfCurrentLayer])):
                            temporalRow.append(currentVector_w[currentWeight][0]-learningRate*Detotal_Dw[currentWeight][0])
                        temporalMatrixOfMatrix_w.append(temporalRow)
                        Detotal_DFzy.append(Detotal_Dfz_TIMES_Dfz_Df)
                    else:
                        fixed_Detotal_DFzy = []
                        for row in range(0, numberOfNeuronsInFinalLayer):
                            fixed_Detotal_DFzy.append(Detotal_DFzy[numberOfNeuronsInFinalLayer-row-1])
                        # ----- We Now update the weights of the other neurons ----- #
                        Detotal_Dfz = [] # = predicted - expected
                        Df_Dw = []
                        Detotal_Dw = [] # = (Detotal_Dfz)*(Dfz_Df)*(Df_Dw)
                        # We calculate "Detotal_Dfz"
                        # trueCurrentLayer
                        Detotal_Dfz = Detotal_DFzy 
                        
                        # We create a temporal matrix for "currentMatrix_w" and "matrix_dFz"
                        # to just re-arrange the structure of how both matrixes
                        # have their actual data. This is needed to then use the
                        # method "self.getCurrentDetotal_DFz()" and to get
                        # Detotal_Dfz through such method.
                        temporalMatrix_w = []
                        temporalDfzMatrix = []
                        
                        # Este loop se repite N veces = "penultima layer" - "1 layer adelante de la actual"
                        for currentFurtherLayer in range(trueCurrentLayer, totalNumberOfLayers):
                            # "temporalNeuronsAnalized" tiene el numero de neuronas que hay en todas las layers anteriores a la actual
                            if (len(temporalMatrix_w) == 0):
                                temporalRow = []
                                temporalNeuronsAnalized = 0
                                for n in range(0, trueCurrentLayer-1):
                                    temporalNeuronsAnalized = temporalNeuronsAnalized + numberOfNeuronsPerLayer[0][n]
                                # We get "Dfz_Df" of the neuron that will improve its weights
                                Dfz_Df = dFz[temporalNeuronsAnalized+trueCurrentNeuronOfCurrentLayer] # = dFz
                                
                                # We get the weights of the neuron that will improve its weights
                                currentVector_w = currentMatrix_w[temporalNeuronsAnalized+trueCurrentNeuronOfCurrentLayer]
                                currentVector_w = matrixMath.getTransposedMatrix([currentVector_w])
                                # We plug in all the weight vectors
                                temporalRow.append(currentVector_w)
                                temporalMatrix_w.append(temporalRow)
                                # We plug in the dFz of the last neuron
                                temporalRow = []
                                currentVector_w = dFz[temporalNeuronsAnalized+trueCurrentNeuronOfCurrentLayer]
                                currentVector_w = matrixMath.getTransposedMatrix([currentVector_w])
                                # We plug in all the weight vectors
                                temporalRow.append(currentVector_w)
                                temporalDfzMatrix.append(temporalRow)
                                
                                # dFz de donde viene la actual weight o del independent variable en caso de tratarse de la 1ra layer
                                # We calculate "Df_Dw"
                                if (trueCurrentLayer == 1):
                                    Df_Dw = matrixMath.getTransposedMatrix(matrix_x)
                                else:
                                    temporalNeuronsAnalized = 0
                                    for n in range(0, trueCurrentLayer-2):
                                        temporalNeuronsAnalized = temporalNeuronsAnalized + numberOfNeuronsPerLayer[0][n]
                                    temporalRow = []
                                    for currentBiasDerivate in range(0, numberOfIndependentRows):
                                        temporalRow.append(1) # bias derivate
                                    Df_Dw.append(temporalRow)
                                    for currentPastNeuronOfPastLayer in range(0, numberOfNeuronsPerLayer[0][trueCurrentLayer-2]):
                                        Df_Dw.append(dFz[temporalNeuronsAnalized+currentPastNeuronOfPastLayer])
                            
                            temporalRow = []
                            temporalNeuronsAnalized = 0
                            for n in range(0, currentFurtherLayer):
                                temporalNeuronsAnalized = temporalNeuronsAnalized + numberOfNeuronsPerLayer[0][n]
                            # Este loop se repite N veces = numero de neuronas en la actual layer (la cual empieza a partir de la layer futura / posterior)
                            for currentFutherNeuronOfCurrentLayer in range(0, numberOfNeuronsPerLayer[0][currentFurtherLayer]):
                                currentVector_w = currentMatrix_w[temporalNeuronsAnalized+currentFutherNeuronOfCurrentLayer]
                                currentVector_w = matrixMath.getTransposedMatrix([currentVector_w])
                                # We plug in all the weight vectors
                                temporalRow.append(currentVector_w)
                            temporalMatrix_w.append(temporalRow)
                            temporalRow = []
                            # Este loop se repite N veces = numero de neuronas en la actual layer (la cual empieza a partir de la layer futura / posterior)
                            for currentFutherNeuronOfCurrentLayer in range(0, numberOfNeuronsPerLayer[0][currentFurtherLayer]):
                                currentVector_w = dFz[temporalNeuronsAnalized+currentFutherNeuronOfCurrentLayer]
                                currentVector_w = matrixMath.getTransposedMatrix([currentVector_w])
                                # We plug in all the weight vectors
                                temporalRow.append(currentVector_w)
                            temporalDfzMatrix.append(temporalRow)
                            
                        # Detotal_DFzy[DerivateOfsample_n][column=0]
                        Detotal_Dfz = self.getCurrentDetotal_DFz(temporalMatrix_w, temporalDfzMatrix, trueCurrentNeuronOfCurrentLayer, fixed_Detotal_DFzy)
                        
                        # We calculate "Detotal_Dw"
                        Detotal_Dfz_TIMES_Dfz_Df = []
                        for currentSample in range(0, len(Detotal_Dfz)):
                            Detotal_Dfz_TIMES_Dfz_Df.append([Detotal_Dfz[currentSample][0] * Dfz_Df[currentSample]])
                        Detotal_Dw = matrixMath.getDotProduct(Df_Dw, Detotal_Dfz_TIMES_Dfz_Df)
                        
                        # We finnally update the weight values of the last neuron
                        # temporalMatrixOfMatrix_w = []
                        temporalNeuronsAnalized = 0
                        for n in range(0, trueCurrentLayer-1):
                            temporalNeuronsAnalized = temporalNeuronsAnalized + numberOfNeuronsPerLayer[0][n]
                        # We get the weights of the neuron that will improve its weights
                        currentVector_w = currentMatrix_w[temporalNeuronsAnalized+trueCurrentNeuronOfCurrentLayer]
                        currentVector_w = matrixMath.getTransposedMatrix([currentVector_w])
                        temporalRow = []
                        for currentWeight in range(0, len(currentMatrix_w[temporalNeuronsAnalized+trueCurrentNeuronOfCurrentLayer])):
                            temporalRow.append(currentVector_w[currentWeight][0]-learningRate*Detotal_Dw[currentWeight][0])
                        temporalMatrixOfMatrix_w.append(temporalRow)
                              
            # We reorder the new obtained weights but accordingly to the neurons
            # order (from neuron 1 to neuron "N") in variable "newMatrix_w"
            newMatrix_w = []
            for row in range(0, totalNumberOfNeurons):
                newMatrix_w.append(temporalMatrixOfMatrix_w[totalNumberOfNeurons-row-1])
                
            # ----- We save the current weight vector performance ----- #
            Fx = []
            Fz = []
            actualFunctionActivation = 0
            for currentLayer in range(0, totalNumberOfLayers):
                temporalRow1 = []
                if (currentLayer == 0):
                    for currentNeuronOfCurrentLayer in range(0, numberOfNeuronsPerLayer[0][currentLayer]):
                        temporalRow1 = matrixMath.getDotProduct(matrix_x, self.getANNweightVectorForOneNeuron(currentMatrix_w, currentNeuronOfCurrentLayer))
                        temporalRow1 = matrixMath.getTransposedMatrix(temporalRow1)
                        Fx.append(temporalRow1)
                    for currentNeuronOfCurrentLayer in range(0, numberOfNeuronsPerLayer[0][currentLayer]):
                        temporalRow1 = []
                        for column in range(0, numberOfIndependentRows):
                            # Activation Functions (Fz)
                            if (activationFunctionsList[0][actualFunctionActivation] == 'none'):
                                current_Fz = Fx[currentNeuronOfCurrentLayer][0][column]
                            if (activationFunctionsList[0][actualFunctionActivation] == 'sigmoid'):
                                current_Fz = self.getSigmoidActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'relu'):
                                current_Fz = self.getReluActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'tanh'):
                                current_Fz = self.getTanhActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo2ndPower'):
                                current_Fz = self.getRaiseToTheSecondPowerActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo3rdPower'):
                                current_Fz = self.getRaiseToTheThirdPowerActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo4thPower'):
                                current_Fz = self.getRaiseToTheFourthPowerActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo5thPower'):
                                current_Fz = self.getRaiseToTheFifthPowerActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo6thPower'):
                                current_Fz = self.getRaiseToTheSixthPowerActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'exponential'):
                                current_Fz = self.getExponentialActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                            temporalRow1.append(current_Fz)
                        actualFunctionActivation = actualFunctionActivation + 1
                        Fz.append(temporalRow1)
                else:
                    pastNeuronOfCurrentLayer = 0
                    for currentLayerCount in range(0, currentLayer-1):
                        pastNeuronOfCurrentLayer = pastNeuronOfCurrentLayer + numberOfNeuronsPerLayer[0][currentLayerCount]
                    inputMatrix = []
                    for row in range(0, numberOfIndependentRows):
                        temporalRow1 = []
                        temporalRow1.append(1) # bias column
                        for currentNeuron in range(pastNeuronOfCurrentLayer, pastNeuronOfCurrentLayer+numberOfNeuronsPerLayer[0][currentLayer-1]):
                            temporalRow1.append(Fz[currentNeuron][row])
                        inputMatrix.append(temporalRow1)
                    for currentNeuronOfCurrentLayer in range(pastNeuronOfCurrentLayer+numberOfNeuronsPerLayer[0][currentLayer-1], pastNeuronOfCurrentLayer+numberOfNeuronsPerLayer[0][currentLayer-1]+numberOfNeuronsPerLayer[0][currentLayer]):
                        temporalRow1 = matrixMath.getDotProduct(inputMatrix, self.getANNweightVectorForOneNeuron(currentMatrix_w, currentNeuronOfCurrentLayer))
                        temporalRow1 = matrixMath.getTransposedMatrix(temporalRow1)
                        Fx.append(temporalRow1)
                    pastNeuronOfCurrentLayer = 0
                    for currentLayerCount in range(0, currentLayer):
                        pastNeuronOfCurrentLayer = pastNeuronOfCurrentLayer + numberOfNeuronsPerLayer[0][currentLayerCount]
                    for currentNeuronOfCurrentLayer in range(0, numberOfNeuronsPerLayer[0][currentLayer]):
                        temporalRow1= []
                        for column in range(0, numberOfIndependentRows):
                            # Activation Functions (Fz)
                            if (activationFunctionsList[0][actualFunctionActivation] == 'none'):
                                current_Fz = Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column]
                            if (activationFunctionsList[0][actualFunctionActivation] == 'sigmoid'):
                                current_Fz = self.getSigmoidActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'relu'):
                                current_Fz = self.getReluActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'tanh'):
                                current_Fz = self.getTanhActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo2ndPower'):
                                current_Fz = self.getRaiseToTheSecondPowerActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo3rdPower'):
                                current_Fz = self.getRaiseToTheThirdPowerActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo4thPower'):
                                current_Fz = self.getRaiseToTheFourthPowerActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo5thPower'):
                                current_Fz = self.getRaiseToTheFifthPowerActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo6thPower'):
                                current_Fz = self.getRaiseToTheSixthPowerActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                            if (activationFunctionsList[0][actualFunctionActivation] == 'exponential'):
                                current_Fz = self.getExponentialActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                            temporalRow1.append(current_Fz)
                        actualFunctionActivation = actualFunctionActivation + 1
                        Fz.append(temporalRow1)
            
            # We evaluate the performance of the innitialized weight vectors
            predictionAcurracy = 0
            predictedData = []
            for currentNeuronOfLastLayer in range(0, numberOfNeuronsInFinalLayer):
                predictedData.append(Fz[totalNumberOfNeurons-numberOfNeuronsInFinalLayer+currentNeuronOfLastLayer])
            predictedData = matrixMath.getTransposedMatrix(predictedData)
            numberOfDataPoints = numberOfIndependentRows*numberOfNeuronsInFinalLayer
            for currentNeuronOfLastLayer in range(0, numberOfNeuronsInFinalLayer):
                for row in range(0, numberOfIndependentRows):
                    n2 = self.y_samplesList[row][currentNeuronOfLastLayer]
                    n1 = predictedData[row][currentNeuronOfLastLayer]
                    if (isClassification == False):
                        if (((n1*n2) != 0)):
                        #if (((n1*n2) > 0) and (n1!=n2)):
                            newAcurracyValueToAdd = (1-(abs(n2-n1)/abs(n2)))
                            if (newAcurracyValueToAdd < 0):
                                newAcurracyValueToAdd = 0
                            predictionAcurracy = predictionAcurracy + newAcurracyValueToAdd
                            #predictionAcurracy = predictionAcurracy + (n1/n2)
                    if (isClassification == True):
                        if (abs(n1) > abs(n2)): # n2 has to be the one with the highest value with respect to n1
                            n2 = predictedData[row][currentNeuronOfLastLayer]
                            n1 = self.y_samplesList[row][currentNeuronOfLastLayer]
                        if ((n1==0) and (n2>=-1 and n2<=1) and (n2!=0)):
                            predictionAcurracy = predictionAcurracy + ((1-abs(n2))/(1-n1))
                        if (n1==n2):
                        #if ((n1==n2) and (n1==0)):
                            predictionAcurracy = predictionAcurracy + 1
            predictionAcurracy = predictionAcurracy/numberOfDataPoints*100
            temporalRow = []
            temporalRow.append(predictionAcurracy)
            temporalRow.append(newMatrix_w)
            allAccuracies.append(temporalRow)
            # We save the current the modeling results if they are better than
            # the actual best
            currentBestAccuracy = bestModelingResults[1]
            if (predictionAcurracy > currentBestAccuracy):
                bestModelingResults = []
                bestModelingResults.append(newMatrix_w)
                bestModelingResults.append(predictionAcurracy)
                bestModelingResults.append(predictedData)
                bestModelingResults.append(firstMatrix_w)
                bestModelingResults.append("Coefficients distribution is as follows:\nmodelCoefficients =\n[\n  [Neuron1_bias, Neuron1_weight1, Neuron1_weight2, ... , Neuron1_weightM],\n  [Neuron2_bias, Neuron2_weight1, Neuron2_weight2, ... , Neuron2_weightZ],\n  [     .      ,        .       ,        .       , ... ,        .       ],\n  [     .      ,        .       ,        .       , ... ,        .       ],\n  [     .      ,        .       ,        .       , ... ,        .       ],\n  [NeuronN_bias, NeuronN_weight1, NeuronN_weight2, ... , NeuronN_weightK],\n]\n")
            if (predictionAcurracy > stopTrainingIfAcurracy):
                break
        # Alongside the information of the best model obtained, we add the
        # modeled information of ALL the models obtained to the variable that
        # we will return in this method
        bestModelingResults.append(allAccuracies)
        return bestModelingResults
    
    """
    predictSingleArtificialNeuron(coefficients="We give the Linear Logistic mathematical coefficients that we want to predict with",
                                  activationFunction="the literal name, in lowercaps, of the activation function that you want to apply the neuron",
                                  isThreshold="Set to True if you want to predict output values of a classification neuron. False if otherwise."
                                  threshold="We give a value from 0 to 1 to indicate the threshold that we want to apply to classify the predicted data with the Linear Logistic Classifier")
    
    This method returns the predicting values of the independent input values
    that you assign in the local variable of this class: "self.x_samplesList".
    The prediction will be made accordingly to the coefficients and
    configuration specified in the arguments of this method.
    
    CODE EXAMPLE:
        # matrix_y = [expectedResult]
        matrix_y = [
                [25.5],
                [31.2],
                [25.9],
                [38.4],
                [18.4],
                [26.7],
                [26.4],
                [25.9],
                [32],
                [25.2],
                [39.7],
                [35.7],
                [26.5]
                ]
        # matrix_x = [variable1, variable2, variable3]
        matrix_x = [
                [1.74, 5.3, 10.8],
                [6.32, 5.42, 9.4],
                [6.22, 8.41, 7.2],
                [10.52, 4.63, 8.5],
                [1.19, 11.6, 9.4],
                [1.22, 5.85, 9.9],
                [4.1, 6.62, 8],
                [6.32, 8.72, 9.1],
                [4.08, 4.42, 8.7],
                [4.15, 7.6, 9.2],
                [10.15, 4.83, 9.4],
                [1.72, 3.12, 7.6],
                [1.7, 5.3, 8.2]
                ]
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        dL = mSL.DeepLearning(matrix_x, matrix_y)
        modelingResults = dL.getSingleArtificialNeuron(activationFunction='none', learningRate=0.001, numberOfEpochs=100000, stopTrainingIfAcurracy=99.9, isCustomizedInitialWeights=False, firstMatrix_w=[], isClassification=False)
        modelCoefficients = modelingResults[0]
        acurracy = modelingResults[1]
        predictedData = modelingResults[2]
        firstMatrix_w = modelingResults[3]
        coefficientDistribution = modelingResults[4]
        allModeledAccuracies = modelingResults[5]
        
        # -------------------------------------------------------- #
        # ----- WE PREDICT SOME DATA WITH OUR CURRENT NEURON ----- #
        # -------------------------------------------------------- #
        matrix_x = [
                [1, 2.3, 3.8],
                [3.32, 2.42, 1.4],
                [2.22, 3.41, 1.2]
                ]
        dL = mSL.DeepLearning(matrix_x, [])
        getPredictedData = dL.predictSingleArtificialNeuron(coefficients=modelCoefficients, activationFunction='none', isThreshold=False, threshold=0.5)
        
        
    EXPECTED CODE RESULT:
        getPredictedData =
        [[28.140432977147068], [28.799532314784063], [25.69562041179361]]
    """
    def predictSingleArtificialNeuron(self, coefficients, activationFunction='sigmoid', isThreshold=True, threshold=0.5):
        if ((activationFunction!='none') and (activationFunction!='sigmoid') and (activationFunction!='relu') and (activationFunction!='tanh') and (activationFunction!='raiseTo2ndPower') and (activationFunction!='raiseTo3rdPower') and (activationFunction!='raiseTo4thPower') and (activationFunction!='raiseTo5thPower') and (activationFunction!='raiseTo6thPower') and (activationFunction!='exponential')):
            raise Exception('ERROR: The selected Activation Function does not exist or has not been programmed in this method yet.')
        from ..linearAlgebra import MortrackLinearAlgebraLibrary as mLAL
        numberOfIndependentRows= len(self.x_samplesList)
        numberOfIndependentVariables = len(self.x_samplesList[0])
        matrix_x = []
        for row in range(0, numberOfIndependentRows):
            temporalRow = []
            temporalRow.append(1)
            for column in range(0, numberOfIndependentVariables):
                temporalRow.append(self.x_samplesList[row][column])
            matrix_x.append(temporalRow)
        matrix_w = coefficients
        # We calculate the results obtained with the given weight coefficients
        # vector
        matrixMath = mLAL.MatrixMath()
        Fx = matrixMath.getDotProduct(matrix_x, matrix_w)
        Fz = []
        if (isThreshold == True):
            for row in range(0, numberOfIndependentRows):
                temporalRow = []
                if (activationFunction == 'none'):
                    current_Fz = Fx[row][0]
                if (activationFunction == 'sigmoid'):
                    current_Fz = self.getSigmoidActivation(Fx[row][0])
                if (activationFunction == 'relu'):
                    current_Fz = self.getReluActivation(Fx[row][0])
                if (activationFunction == 'tanh'):
                    current_Fz = self.getTanhActivation(Fx[row][0])
                if (activationFunction == 'raiseTo2ndPower'):
                    current_Fz = self.getRaiseToTheSecondPowerActivation(Fx[row][0])
                if (activationFunction == 'raiseTo3rdPower'):
                    current_Fz = self.getRaiseToTheThirdPowerActivation(Fx[row][0])
                if (activationFunction == 'raiseTo4thPower'):
                    current_Fz = self.getRaiseToTheFourthPowerActivation(Fx[row][0])
                if (activationFunction == 'raiseTo5thPower'):
                    current_Fz = self.getRaiseToTheFifthPowerActivation(Fx[row][0])
                if (activationFunction == 'raiseTo6thPower'):
                    current_Fz = self.getRaiseToTheSixthPowerActivation(Fx[row][0])
                if (activationFunction == 'exponential'):
                    current_Fz = self.getExponentialActivation(Fx[row][0])
                if (current_Fz < threshold):
                    current_Fz = 0
                else:
                    current_Fz = 1
                temporalRow.append(current_Fz)
                Fz.append(temporalRow)
        else:
            for row in range(0, numberOfIndependentRows):
                temporalRow = []
                if (activationFunction == 'none'):
                    current_Fz = Fx[row][0]
                if (activationFunction == 'sigmoid'):
                    current_Fz = self.getSigmoidActivation(Fx[row][0])
                if (activationFunction == 'relu'):
                    current_Fz = self.getReluActivation(Fx[row][0])
                if (activationFunction == 'tanh'):
                    current_Fz = self.getTanhActivation(Fx[row][0])
                if (activationFunction == 'raiseTo2ndPower'):
                    current_Fz = self.getRaiseToTheSecondPowerActivation(Fx[row][0])
                if (activationFunction == 'raiseTo3rdPower'):
                    current_Fz = self.getRaiseToTheThirdPowerActivation(Fx[row][0])
                if (activationFunction == 'raiseTo4thPower'):
                    current_Fz = self.getRaiseToTheFourthPowerActivation(Fx[row][0])
                if (activationFunction == 'raiseTo5thPower'):
                    current_Fz = self.getRaiseToTheFifthPowerActivation(Fx[row][0])
                if (activationFunction == 'raiseTo6thPower'):
                    current_Fz = self.getRaiseToTheSixthPowerActivation(Fx[row][0])
                if (activationFunction == 'exponential'):
                    current_Fz = self.getExponentialActivation(Fx[row][0])
                temporalRow.append(current_Fz)
                Fz.append(temporalRow)
                
        # We get the predicted Values
        predictedData = Fz
        # We return the predicted data
        return predictedData
    
    """
    predictArtificialNeuralNetwork(coefficients="We give the Linear Logistic mathematical coefficients that we want to predict with",
                                  activationFunction="the literal name, in lowercaps, of the activation function that you want to apply the neuron",
                                  isThreshold="Set to True if you want to predict output values of a classification neuron. False if otherwise."
                                  threshold="We give a value from 0 to 1 to indicate the threshold that we want to apply to classify the predicted data with the Linear Logistic Classifier")
    
    This method returns the predicting values of the independent input values
    that you assign in the local variable of this class: "self.x_samplesList".
    The prediction will be made accordingly to the coefficients and
    configuration specified in the arguments of this method.
    
    CODE EXAMPLE:
        # matrix_y = [expectedResult]
        matrix_y = [
                [25.5],
                [31.2],
                [25.9],
                [38.4],
                [18.4],
                [26.7],
                [26.4],
                [25.9],
                [32],
                [25.2],
                [39.7],
                [35.7],
                [26.5]
                ]
        # matrix_x = [variable1, variable2, variable3]
        matrix_x = [
                [1.74, 5.3, 10.8],
                [6.32, 5.42, 9.4],
                [6.22, 8.41, 7.2],
                [10.52, 4.63, 8.5],
                [1.19, 11.6, 9.4],
                [1.22, 5.85, 9.9],
                [4.1, 6.62, 8],
                [6.32, 8.72, 9.1],
                [4.08, 4.42, 8.7],
                [4.15, 7.6, 9.2],
                [10.15, 4.83, 9.4],
                [1.72, 3.12, 7.6],
                [1.7, 5.3, 8.2]
                ]
        from MortrackAPI.machineLearning import MortrackML_Library as mSL
        dL = mSL.DeepLearning(matrix_x, matrix_y)
        # We will indicate that we want 2 neurons in Layer1 and 1 neuron in Layer2
        aNND = [
                [1,1,1],
                [0,1,0]
                ]
        aF = [
              ['none', 'none', 'none'],
              ['', 'none', '']
              ]
        modelingResults = dL.getArtificialNeuralNetwork(artificialNeuralNetworkDistribution=aNND, activationFunction=aF, learningRate=0.00001, numberOfEpochs=100000, stopTrainingIfAcurracy=99.9, isCustomizedInitialWeights=False, firstMatrix_w=[], isClassification=False)
        modelCoefficients = modelingResults[0]
        acurracy = modelingResults[1]
        predictedData = modelingResults[2]
        firstMatrix_w = modelingResults[3]
        coefficientDistribution = modelingResults[4]
        allModeledAccuracies = modelingResults[5]
        
        # -------------------------------------------------------- #
        # ----- WE PREDICT SOME DATA WITH OUR CURRENT NEURON ----- #
        # -------------------------------------------------------- #
        matrix_x = [
                [1, 2.3, 3.8],
                [3.32, 2.42, 1.4],
                [2.22, 3.41, 1.2]
                ]
        # We will indicate that we want 2 neurons in Layer1 and 1 neuron in Layer2
        aNND = [
                [1,1,1],
                [0,1,0]
                ]
        aF = [
              ['none', 'none', 'none'],
              ['', 'none', '']
              ]
        dL = mSL.DeepLearning(matrix_x, [])
        getPredictedData = dL.predictArtificialNeuralNetwork(coefficients=modelCoefficients, artificialNeuralNetworkDistribution=aNND, activationFunction=aF, isThreshold=False, threshold=0.5)
        
        
    EXPECTED CODE RESULT:
        getPredictedData =
        [[28.22084819611869], [28.895166544625255], [25.788001189515317]]
    """
    def predictArtificialNeuralNetwork(self, coefficients, artificialNeuralNetworkDistribution, activationFunction, isThreshold=True, threshold=0.5):
        from ..linearAlgebra import MortrackLinearAlgebraLibrary as mLAL
        from . import MortrackML_Library as mSL
        numberOfIndependentRows= len(self.x_samplesList)
        numberOfIndependentVariables = len(self.x_samplesList[0])
        numberOfNeuronLayers = len(artificialNeuralNetworkDistribution[0])
        numberOfNeuronsPerLayer = []
        activationFunctionsList = []
        totalNumberOfNeurons = 0
        matrixMath = mLAL.MatrixMath()
        transposedANND = matrixMath.getTransposedMatrix(artificialNeuralNetworkDistribution)
        transposedAF = matrixMath.getTransposedMatrix(activationFunction)
        for row in range(0, len(transposedANND)):
            currentNumberOfNeurons = 0
            for column in range(0, len(transposedANND[0])):
                if (transposedANND[row][column] == 1):
                    currentNumberOfNeurons = currentNumberOfNeurons + 1
                    activationFunctionsList.append(transposedAF[row][column])
            temporalRow = []
            temporalRow.append(currentNumberOfNeurons)
            numberOfNeuronsPerLayer.append(temporalRow)
            totalNumberOfNeurons = totalNumberOfNeurons + currentNumberOfNeurons
        numberOfNeuronsPerLayer = matrixMath.getTransposedMatrix(numberOfNeuronsPerLayer)
        activationFunctionsList = [activationFunctionsList]
        numberOfNeuronsInFinalLayer = numberOfNeuronsPerLayer[0][len(numberOfNeuronsPerLayer[0])-1]
        for column in range(0, numberOfNeuronLayers):
            for row in range(0, numberOfNeuronsPerLayer[0][column]):
                if ((activationFunction[row][column]!='none') and (activationFunction[row][column]!='sigmoid') and (activationFunction[row][column]!='relu') and (activationFunction[row][column]!='tanh') and (activationFunction[row][column]!='raiseTo2ndPower') and (activationFunction[row][column]!='raiseTo3rdPower') and (activationFunction[row][column]!='raiseTo4thPower') and (activationFunction[row][column]!='raiseTo5thPower') and (activationFunction[row][column]!='raiseTo6thPower') and (activationFunction[row][column]!='exponential')):
                    raise Exception('ERROR: The selected Activation Function does not exist or has not been programmed in this method yet.')
        totalNumberOfLayers = len(numberOfNeuronsPerLayer[0])
        matrix_x = []
        for row in range(0, numberOfIndependentRows):
            temporalRow = []
            temporalRow.append(1)
            for column in range(0, numberOfIndependentVariables):
                temporalRow.append(self.x_samplesList[row][column])
            matrix_x.append(temporalRow)
                
        # ----- Predict the output values with the given weight matrix ----- #
        currentMatrix_w = coefficients
        Fx = []
        Fz = []
        actualFunctionActivation = 0
        for currentLayer in range(0, totalNumberOfLayers):
            temporalRow1 = []
            if (currentLayer == 0):
                for currentNeuronOfCurrentLayer in range(0, numberOfNeuronsPerLayer[0][currentLayer]):
                    temporalRow1 = matrixMath.getDotProduct(matrix_x, self.getANNweightVectorForOneNeuron(currentMatrix_w, currentNeuronOfCurrentLayer))
                    temporalRow1 = matrixMath.getTransposedMatrix(temporalRow1)
                    Fx.append(temporalRow1)
                for currentNeuronOfCurrentLayer in range(0, numberOfNeuronsPerLayer[0][currentLayer]):
                    temporalRow1 = []
                    for column in range(0, numberOfIndependentRows):
                        # Activation Functions (Fz)
                        if (activationFunctionsList[0][actualFunctionActivation] == 'none'):
                            current_Fz = Fx[currentNeuronOfCurrentLayer][0][column]
                        if (activationFunctionsList[0][actualFunctionActivation] == 'sigmoid'):
                            current_Fz = self.getSigmoidActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'relu'):
                            current_Fz = self.getReluActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'tanh'):
                            current_Fz = self.getTanhActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo2ndPower'):
                            current_Fz = self.getRaiseToTheSecondPowerActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo3rdPower'):
                            current_Fz = self.getRaiseToTheThirdPowerActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo4thPower'):
                            current_Fz = self.getRaiseToTheFourthPowerActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo5thPower'):
                            current_Fz = self.getRaiseToTheFifthPowerActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo6thPower'):
                            current_Fz = self.getRaiseToTheSixthPowerActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'exponential'):
                            current_Fz = self.getExponentialActivation(Fx[currentNeuronOfCurrentLayer][0][column])
                        temporalRow1.append(current_Fz)
                    actualFunctionActivation = actualFunctionActivation + 1
                    Fz.append(temporalRow1)
            else:
                pastNeuronOfCurrentLayer = 0
                for currentLayerCount in range(0, currentLayer-1):
                    pastNeuronOfCurrentLayer = pastNeuronOfCurrentLayer + numberOfNeuronsPerLayer[0][currentLayerCount]
                inputMatrix = []
                for row in range(0, numberOfIndependentRows):
                    temporalRow1 = []
                    temporalRow1.append(1) # bias column
                    for currentNeuron in range(pastNeuronOfCurrentLayer, pastNeuronOfCurrentLayer+numberOfNeuronsPerLayer[0][currentLayer-1]):
                        temporalRow1.append(Fz[currentNeuron][row])
                    inputMatrix.append(temporalRow1)
                for currentNeuronOfCurrentLayer in range(pastNeuronOfCurrentLayer+numberOfNeuronsPerLayer[0][currentLayer-1], pastNeuronOfCurrentLayer+numberOfNeuronsPerLayer[0][currentLayer-1]+numberOfNeuronsPerLayer[0][currentLayer]):
                    temporalRow1 = matrixMath.getDotProduct(inputMatrix, self.getANNweightVectorForOneNeuron(currentMatrix_w, currentNeuronOfCurrentLayer))
                    temporalRow1 = matrixMath.getTransposedMatrix(temporalRow1)
                    Fx.append(temporalRow1)
                pastNeuronOfCurrentLayer = 0
                for currentLayerCount in range(0, currentLayer):
                    pastNeuronOfCurrentLayer = pastNeuronOfCurrentLayer + numberOfNeuronsPerLayer[0][currentLayerCount]
                for currentNeuronOfCurrentLayer in range(0, numberOfNeuronsPerLayer[0][currentLayer]):
                    temporalRow1= []
                    for column in range(0, numberOfIndependentRows):
                        # Activation Functions (Fz)
                        if (activationFunctionsList[0][actualFunctionActivation] == 'none'):
                            current_Fz = Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column]
                        if (activationFunctionsList[0][actualFunctionActivation] == 'sigmoid'):
                            current_Fz = self.getSigmoidActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'relu'):
                            current_Fz = self.getReluActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'tanh'):
                            current_Fz = self.getTanhActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo2ndPower'):
                            current_Fz = self.getRaiseToTheSecondPowerActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo3rdPower'):
                            current_Fz = self.getRaiseToTheThirdPowerActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo4thPower'):
                            current_Fz = self.getRaiseToTheFourthPowerActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo5thPower'):
                            current_Fz = self.getRaiseToTheFifthPowerActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'raiseTo6thPower'):
                            current_Fz = self.getRaiseToTheSixthPowerActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                        if (activationFunctionsList[0][actualFunctionActivation] == 'exponential'):
                            current_Fz = self.getExponentialActivation(Fx[currentNeuronOfCurrentLayer+pastNeuronOfCurrentLayer][0][column])
                        temporalRow1.append(current_Fz)
                    actualFunctionActivation = actualFunctionActivation + 1
                    Fz.append(temporalRow1)
        
        # We get the predicted values and we then give them the proper
        # row-column format to then return it.
        pastNeuronOfCurrentLayer = 0
        for currentLayerCount in range(0, numberOfNeuronLayers-1):
            pastNeuronOfCurrentLayer = pastNeuronOfCurrentLayer + numberOfNeuronsPerLayer[0][currentLayerCount]
        prePredictedValues = []
        for currentNeuronOfFinalLayer in range(0, numberOfNeuronsInFinalLayer):
            temporalNeuron = []
            for row in range(0, len(Fz[pastNeuronOfCurrentLayer + currentNeuronOfFinalLayer])):
                temporalRow = []
                current_Fz = Fz[pastNeuronOfCurrentLayer + currentNeuronOfFinalLayer][row]
                if (isThreshold == True):
                    if (current_Fz < threshold):
                        current_Fz = 0
                    else:
                        current_Fz = 1
                    temporalRow.append([current_Fz])
                else:
                    temporalRow.append([Fz[pastNeuronOfCurrentLayer + currentNeuronOfFinalLayer][row]])
                temporalNeuron.append(temporalRow)
            prePredictedValues.append(temporalNeuron)
        predictedValues = []
        for currentSample in range(0, numberOfIndependentRows):
            temporalRow = []
            for currentNeuronOfFinalLayer in range(0, len(prePredictedValues)):
                temporalRow.append(prePredictedValues[currentNeuronOfFinalLayer][currentSample][0][0])
            predictedValues.append(temporalRow)
        return predictedValues
    