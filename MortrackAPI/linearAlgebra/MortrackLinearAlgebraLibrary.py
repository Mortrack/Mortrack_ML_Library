
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

# IMPORTANT NOTE: Know that for this class "MatrixMath", matrixes are expected
#                 to be ordered as ("row", "column")
# IMPORTANT NOTE: Remember to be careful with input argument variables when
#                 you call any method or class because it gets altered in
#                 python ignoring parenting or children logic
class MatrixMath:
    #def __init__(self):
    
    """
    IMPORTANT NOTE: THIS METHOD REQUIRES APROVAL!!!
    
    getGaussJordanElimination("matrix x", "matrix y")
    
    This method solves the equation expressed in the matrix "x" (dependent
    variable) and the matrix "y" (the independent variable). It is expected
    that the matrix "y" is distributed in a 1 single column matrix with several
    rows, just like in the traditional math representation in most books.
    
    CODE EXAMPLE:
        
    EXPECTED RESULT:
    """
    def getGaussJordanElimination(self, matrix_x, matrix_y):
        return matrix_x
        matrixLength = len(matrix_x)
        rowMultiplier = 0
        # ----- We get the Inverse of the Matrix through Gauss Method ----- #
        nextUnitaryValue = [0, 0]
        for column in range(0, matrixLength):
            # We make unitary the corresponding value of the Actual Column
            if (matrix_x[nextUnitaryValue[0]][nextUnitaryValue[1]] == 0):
                rowSelected = 0
                for n in range(0, matrixLength):
                    if (matrix_x[nextUnitaryValue[0]][n] != 0):
                        rowSelected = n
                        break
                for n in range(0, matrixLength):
                    matrix_x[nextUnitaryValue[0]][n] = matrix_x[nextUnitaryValue[0]][n] + matrix_x[rowSelected][n]
                matrix_y[nextUnitaryValue[0]][0] = matrix_y[nextUnitaryValue[0]][0] + matrix_y[rowSelected][0]
            else:
                rowMultiplier = 1/matrix_x[nextUnitaryValue[0]][nextUnitaryValue[1]]
                for n in range(0, matrixLength):
                    matrix_x[nextUnitaryValue[0]][n] = rowMultiplier*matrix_x[nextUnitaryValue[0]][n]
                matrix_y[nextUnitaryValue[0]][0] = rowMultiplier*matrix_y[nextUnitaryValue[0]][0]
            # We make the other values of Actual Column equal to zero
            for row in range(0, matrixLength):
                if (row != nextUnitaryValue[0]):
                    if (matrix_x[row][nextUnitaryValue[1]] != 0):
                        rowMultiplier = matrix_x[row][nextUnitaryValue[1]]
                    for n in range(0, matrixLength):
                        matrix_x[row][n] = matrix_x[row][n] - rowMultiplier*matrix_x[nextUnitaryValue[0]][n]
                    matrix_y[row][0] = matrix_y[row][0] - rowMultiplier*matrix_y[nextUnitaryValue[0]][0]
            nextUnitaryValue[0] = nextUnitaryValue[0] + 1
            nextUnitaryValue[1] = nextUnitaryValue[1] + 1
        return matrix_x
    
    """
    getMultiplication("matrix A", "matrix B")
    
    This method multiplies matrix "A" with matrix "B", were matrix "A" is
    considered the first matrix and matrix "B" is considered the second one.
    
    CODE EXAMPLE:
        from MortrackLibrary.linearAlgebra import MortrackLinearAlgebraLibrary as mLAL
        matrixMath = mLAL.MatrixMath()
        matrix1 = [[-2, 3], [-5, 1], [0, -6]]
        matrix2 = [[1, -5, 0], [-8, 9, 2]]
        resultOfMultiplication = matrixMath.getMultiplication(matrix1, matrix2)
        
    EXPECTED RESULT:
        resultOfMultiplication =
        [[-26, 37, 6], [-13, 34, 2], [48, -54, -12]]
    """
    def getMultiplication(self, matrix_A, matrix_B):
        if (len(matrix_A[0]) != len(matrix_B)):
            raise Exception('ERROR: The number of columns of the first matrix does not match the number of rows of the second matrix.')
        matrix_Result = []
        for n in range(0, len(matrix_A)):
            temporalRow = []
            for i in range(0, len(matrix_B[0])):
                temporalSum = 0
                for j in range(0, len(matrix_A[0])):
                    temporalSum = temporalSum + matrix_A[n][j]*matrix_B[j][i]
                temporalRow.append(temporalSum)
            matrix_Result.append(temporalRow)
        return matrix_Result
    
    """
    getDotProduct("matrix A", "matrix B")
    
    This method gets Dot Product between matrix "A" with matrix "B", were
    matrix "A" is considered the first matrix and matrix "B" is considered the
    second one.
    
    CODE EXAMPLE:
        from MortrackLibrary.linearAlgebra import MortrackLinearAlgebraLibrary as mLAL
        matrixMath = mLAL.MatrixMath()
        matrix1 = [[1, 2, 1], [3, 4, 1]]
        matrix2 = [[5, 6], [7, 8], [9, 10]]
        resultOfDotProduct = matrixMath.getDotProduct(matrix1, matrix2)
        
    EXPECTED RESULT:
        resultOfDotProduct =
        [[28, 32], [52, 60]]
    """
    def getDotProduct(self, matrix_A, matrix_B):
        if (len(matrix_A[0]) != len(matrix_B)):
            raise Exception('ERROR: The number of columns of the first matrix does not match the number of rows of the second matrix.')
        matrix_Result = []
        for row in range(0, len(matrix_A)):
            temporalRow = []
            for column in range(0, len(matrix_B[0])):
                totalSum = 0
                for currentDotProduct in range(0, len(matrix_B)):
                    totalSum = totalSum + matrix_A[row][currentDotProduct]*matrix_B[currentDotProduct][column]
                temporalRow.append(totalSum)
            matrix_Result.append(temporalRow)
        return matrix_Result
    
    """
    getInverse("matrix that you want to get the inverse from")
    
    This method calculates the inverse matrix of the inputed matrix.
    
    CODE EXAMPLE:
        from MortrackLibrary.linearAlgebra import MortrackLinearAlgebraLibrary as mLAL
        matrixMath = mLAL.MatrixMath()
        matrix_x = [[1, 2], [3, 4]]
        inversedMatrix_x = matrixMath.getInverse(matrix_x)
        
    EXPECTED RESULT:
        inversedMatrix_x = 
        [[-2.0, 1.0], [1.5, -0.5]]
    """
    def getInverse(self, matrixList):
        matrixLength = len(matrixList)
        mainMatrix = matrixList
        # We create a unitary matrix with the same length as the one we want to
        # get its inverse matrix
        inverseMatrix = self.getUnitaryMatrix(matrixLength)
        rowMultiplier = 0
        # ----- We get the Inverse of the Matrix through Gauss Method ----- #
        nextUnitaryValue = [0, 0]
        for column in range(0, matrixLength):
            # We make unitary the corresponding value of the Actual Column
            if (mainMatrix[nextUnitaryValue[0]][nextUnitaryValue[1]] == 0):
                rowSelected = 0
                for n in range(0, matrixLength):
                    if (mainMatrix[nextUnitaryValue[0]][n] != 0):
                        rowSelected = n
                        break
                for n in range(0, matrixLength):
                    mainMatrix[nextUnitaryValue[0]][n] = mainMatrix[nextUnitaryValue[0]][n] + mainMatrix[rowSelected][n]
                    inverseMatrix[nextUnitaryValue[0]][n] = inverseMatrix[nextUnitaryValue[0]][n] + inverseMatrix[rowSelected][n]
            else:
                rowMultiplier = 1/mainMatrix[nextUnitaryValue[0]][nextUnitaryValue[1]]
                for n in range(0, matrixLength):
                    mainMatrix[nextUnitaryValue[0]][n] = rowMultiplier*mainMatrix[nextUnitaryValue[0]][n]
                    inverseMatrix[nextUnitaryValue[0]][n] = rowMultiplier*inverseMatrix[nextUnitaryValue[0]][n]
            # We make the other values of Actual Column equal to zero
            for row in range(0, matrixLength):
                if (row != nextUnitaryValue[0]):
                    if (mainMatrix[row][nextUnitaryValue[1]] != 0):
                        rowMultiplier = mainMatrix[row][nextUnitaryValue[1]]
                    for n in range(0, matrixLength):
                        mainMatrix[row][n] = mainMatrix[row][n] - rowMultiplier*mainMatrix[nextUnitaryValue[0]][n]
                        inverseMatrix[row][n] = inverseMatrix[row][n] - rowMultiplier*inverseMatrix[nextUnitaryValue[0]][n]
            nextUnitaryValue[0] = nextUnitaryValue[0] + 1
            nextUnitaryValue[1] = nextUnitaryValue[1] + 1
        return inverseMatrix
    
    """
    getTransposedMatrix("matrix that we want to get the transposed matrix from")
    
    This method returns the transposed matrix of the inputed matrix in the
    arguments of this method.
    
    CODE EXAMPLE:
        from MortrackLibrary.linearAlgebra import MortrackLinearAlgebraLibrary as mLAL
        m = [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
        matrixMath = mLAL.MatrixMath()
        transposedMatrix = matrixMath.getTransposedMatrix(m)
        
    EXPECTED RESULT:
        transposedMatrix =
        [[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]]
    """
    def getTransposedMatrix(self, matrixList):
        matrixRowLength = len(matrixList)
        matrixColumnLength = len(matrixList[0])
        transposedMatrix = []
        for originalColumn in range(0, matrixColumnLength):
            temporalRow = []
            for originalRow in range(0, matrixRowLength):
                temporalRow.append(matrixList[originalRow][originalColumn])
            transposedMatrix.append(temporalRow)
        return transposedMatrix
    
    """
    getUnitaryMatrix("desired length in rows / columns that you want for the unitary matrix to create")
    
    This method creates a unitary matrix with a specified length.
    
    CODE EXAMPLE:
        from MortrackLibrary.linearAlgebra import MortrackLinearAlgebraLibrary as mLAL
        matrixMath = mLAL.MatrixMath()
        unitaryMatrix = matrixMath.getUnitaryMatrix(4)

    EXPECTED RESULT:
        unitaryMatrix =
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    """
    def getUnitaryMatrix(self, matrixRowOrColumnLength):
        unitaryMatrix = []
        matrixLength = matrixRowOrColumnLength
        nextUnitaryValue = [0, 0]
        for n in range(0, matrixLength):
            temporalRow = []
            for i in range(0, matrixLength):
                if ((n==nextUnitaryValue[0]) and (i==nextUnitaryValue[1])):
                    nextUnitaryValue[0] = nextUnitaryValue[0] + 1
                    nextUnitaryValue[1] = nextUnitaryValue[1] + 1
                    temporalRow.append(1)
                else:
                    temporalRow.append(0)
            unitaryMatrix.append(temporalRow)
        return unitaryMatrix
            
            
    