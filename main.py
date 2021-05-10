
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# COMPLETITION DATE: May 26, 2020.
# AUTHOR OF THE MORTRACK ML API: Engineer Cesar Miranda Meza (alias: Mortrack).
#
# Hoping that the Mortrack ML API is useful for your projects, the following
# template is proposed by the engineer Cesar Miranda Meza (alias Mortrack) as
# a best practice to work with machine learning algorithms. Enjoy and have fun!
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #



# ------------------------------- #
# ----- IMPORT OF LIBRARIES ----- #
# ------------------------------- #
# "mLAL" is the library that provides some of the mathematical functions of
# linear algebra.
from MortrackAPI.linearAlgebra import MortrackLinearAlgebraLibrary as mLAL
# "mSL" is the library that provides all the algorithms and tools of machine
# learning.
from MortrackAPI.machineLearning import MortrackML_Library as mSL


# ----------------------------- #
# ----- IMPORT OF DATASET ----- #
# ----------------------------- #
# This section of code is dedicated to import the datasets that will be
# required for the machine learning application to solve.


# ------------------------------------- #
# ----- PREPROCESSING OF THE DATA ----- #
# ------------------------------------- #
# This section of code is used to process the imported dataset so that it can
# be properly used by the machine learning algorithms, but only if needed.
# Data preprocessing may include the removal or treatment of missing data in
# the dataset to be worked with. In addition, it may also include encoding of
# categorical data, which consists in using dummy variables in categorical
# input values that were not labeled/defined numerically.


# -------------------------- #
# ----- DATA SPLITTING ----- #
# -------------------------- #
# This section of code is employed to avoid biases in the model that will be
# generated. Despite being an optional process, data here is divided into
# "training and test data" or "into training, test and cross-validation data",
# when a more reliable model is desired.


# --------------------------- #
# ----- FEATURE SCALING ----- #
# --------------------------- #
# This section of code is used to have a big impact on the results of the
# modeling stage. This is because it relies on the fact that this tool
# standarizes/normalizes the input data that will be used to train the
# selected algorithm. Therefore, allowing machine learning algorithms to
# perform better when there are a mix of slow and abrupt changes in the input
# data. However, this whole process is optional to be used.


# ------------------------- #
# ----- DATA MODELING ----- #
# ------------------------- #
# This section of code is dedicated to select a machine learning algorithm
# and then to train it with the resulting training data from the previous
# processes. Subsequently, a validation of the model with the test and/or
# cross validation data can be applied to determine whether the model is good
# or not. As a consequence of this, if the model has not been able to meet
# the desired expectations, the data modeling step can be repeated but with a
# different algorithm.


# ------------------------------------ #
# ----- PREDICTIONS OF THE MODEL ----- #
# ------------------------------------ #
# This section of code is employed to make any desired prediction with the
# model that was generated through the training process.


