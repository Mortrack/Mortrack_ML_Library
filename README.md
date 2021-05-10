# Mortrack_ML_API

## What is Mortrack_ML_API?
This is an API whose creation was started from scratch by engineer Cesar Miranda Meza (alias Mortrack) on May 1, 2020 for personal use at first.
Without any assistance and intuitively using well-known mathematical tools, the author completed this API on May 26, 2020.
It was then first publicly released on May 10, 2021 so that it can help others as helped the author in many of his professional works and research.
In this sense, the Mortrack_ML_API proposes a complete and different approach to working with machine learning algorithms compared other APIs.
It has the outstanding features and philosophy of having lightweight code, having transparent output trained models and providing training means on a different computer from where it is being applied.
Furthermore, this API has been engineering in the Python programming languange and to have a similar framework as the current most popular machine learning APIs in order to provide a user-friendly and intuitive programming interface.

Latest public stable version: 1.0.0.0 (released on May 10, 2021 in master branch)

## What machine learning tools does this API provide?
1. Functions for statistics:
    - Function to calculate the mean in 1 or more lists contained within an array.
    - Function to calculate the variance in 1 or more lists contained within an array.
    - Function to calculate the standard deviation in 1 or more lists contained within an array.
    - Functions to compute confidence intervals in 1 or more lists contained within a matrix.
        - Function to calculate the critical value for confidence intervals of 95%, 99% or 99.9% in 1 or more lists contained within a matrix.
        - Function to calculate the average confidence intervals on 1 or more lists contained within a matrix.
        - Function to compute prediction intervals on 1 or more lists contained within a matrix.
2. Functions for database processing:
    - Function to apply data splitting in a database in a customized way (includes random data splitting option).
3. Functions to apply data scaling:
    - Function to return a dataset with the standardization method applied to it, together with the mean and standard deviation calculated for the application of the method.
    - Function to return a dataset with the inverse application of the standardization method under a value of the mean and standard deviation to be specified when calling this function.
4. Functions for machine learning (all these methods include a function for training and one for making predictions):
    - Functions for regression methods:
        - Linear logistic regression without hyperparameters (gives continuous values, so if a threshold is applied, then this function could be used for classification instead of a regression method).
        - Linear regression without hyperparameters.
        - Multiple linear regression without hyperparameters.
        - Polynomial regression without hyperparameters.
        - Multiple polynomial regression without hyperparameters.
        - Customized multiple second order polynomial regression (y = bo + b1*x1 + b2*x1^2 + b3*x2 + b4*x2^2 + b5*x1*x2).
        -Customized multiple second order polynomial regression (y = bo + b1*x1 + b2*x1^2 + b3*x1^3 + b4*x2 + b5*x2^2 + b6*x2^3 + b7*x1*x2 + b8*x1^2*x2 + b9*x1*x2^2).
    - Functions for classification methods:
        - Support vector machine classification without hyperparameters.
        - Kernel support vector machine classification without hyperparameters.
        - Classification with linear logistic regression without hyperparameters.
    - Functions for reinforcement learning methods:
        - Upper confidence bound with the capacity to learn from an entire complete history.
        - Upper confidence bound with the capacity to learn from real-time data.
        - Modified upper confidence bound with the capacity to learn from an entire complete history (prediction intervals are used within the logic of the algorithm).
        - Modified upper confidence bound with the capacity to learn from real-time data (prediction intervals are used within the logic of the algorithm).
    - Functions for deep learning methods:
        - Single artificial neuron
        - Artificial neural network

## How to use this API?
1. Download it either manually or through git in a terminal window on your computer and consider that it is recommended to pull from the master branch to get the latest stable version. 
2. Open the file named "main.py" located in the root directory of the pulled/downloaded files.
It is recommended as a good practice to program and use this API through that file and follow the proposed machine learning programming framework.
3. Once you have identified that this API contains the machine learning algorithm you wish to use, inspect the file named "MortrackML_Library.py" located in "/MortrackAPI/machineLearning/".
4. In that file, search for the class method that has the name of the algorithm you want to use and you will be able to identify a description of how to use it.
This will appear as a commented section just before the code of that method and there you you will also have at your disposal an example code.
5. Use the code example as a reference to apply it according to your needs in the "main.py" file identified in step 2.

NOTE: As a bonus, this API also some linear algebra mathematical tools that you can find in the file named "MortrackLinearAlgebraLibrary.py" located in "/MortrackAPI/linearAlgebra/".
The documentation on how to use it is also in commented sections within the file just like in the machine learning library file (MortrackML_Library.py).

### Permissions, conditions and limitations to use this API
In accordance to the Apache License 2.0 which this API has, you can use it for commercial use, you can distribute it, modify it and/or use it in privately as long as a copy of its license is included and as long as changes to this API are documented.
It is also requested to give credit to the author (engineer Cesar Miranda Meza, alias Mortrack) and to be aware that this license includes a limitation of liability, explicitly states that it does NOT provide any warranty and it explicitly states that it does NOT grant trademark rights.
