import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from sklearn.metrics import accuracy_score
from neatdata.neatdata import *
from sklearn.metrics import confusion_matrix
import pickle

class ModelPipeline:

    def __init__(self):
        self.indexColumns, self.iWillManuallyCleanColumns = None, None
        self.neatData =  NeatData()
        self.className = 'class' # Edit: Replace class with the Y column name
        self.indexColumns = [] # Edit: Optionally add column names
        self.iWillManuallyCleanColumns = [] # Edit: Optionally add column names
        self.cleanTrainX, self.cleanTrainY, self.cleanTestX, self.cleanTestY = None, None, None, None
        self.results = None


    def execute(self):
        trainX, testX, trainY, testY = self._getDatasetFromOneFile() # Edit: choose one of two functions
        self._cleanDatasets(trainX, testX, trainY, testY)
        self._modelFit()
        self._printModelScores()
        self._createTrainedModelPipelineFile()
        self._saveObjectsToDisk()
        self._createTrainedModelPipelineFile()

    def _getDatasetFromOneFile(self):
        df = pd.read_csv("train.csv", header=None) # Edit: Your dataset
        classDF = pd.read_csv("train_labels.csv", header=None, names=["class"])
        df = pd.concat([df, classDF], axis=1)

        trainX, testX, trainY, testY = train_test_split(df.drop([self.className], axis=1),
                                                         df[self.className], train_size=0.75, test_size=0.25)
        return trainX, testX, trainY, testY

    def _getDatasetFromTwoFiles(self):
        trainingDf = pd.read_csv('train_iris.csv') # Edit: Your training dataset
        testDf = pd.read_csv('test_iris.csv') # Edit: Your test dataset
        trainX = trainingDf.drop([self.className], axis=1)
        trainY = trainingDf[self.className]
        testX = testDf.drop([self.className], axis=1)
        testY = testDf[self.className]
        return trainX, testX, trainY, testY

    def _cleanDatasets(self, trainX, testX, trainY, testY):
        self.cleanTrainX, self.cleanTrainY = self.neatData.cleanTrainingDataset(trainX, trainY, self.indexColumns, self.iWillManuallyCleanColumns)
        self.cleanTestX = self.neatData.cleanTestDataset(testX)
        self.cleanTestY = self.neatData.convertYToNumbersForModeling(testY)

    def _modelFit(self):
        exported_pipeline = make_pipeline(
            StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.1, max_depth=4, max_features=0.2, min_samples_leaf=5, min_samples_split=3, n_estimators=100, subsample=0.6000000000000001)),
            GaussianNB()
        )

        self.exported_pipeline = exported_pipeline
        self.exported_pipeline.fit(self.cleanTrainX, self.cleanTrainY)
        self.results = self.exported_pipeline.predict(self.cleanTestX)

    def _printModelScores(self):
        print("Confusion Matrix:")
        print(confusion_matrix(self.cleanTestY, self.results))
        print(accuracy_score(self.cleanTestY, self.results))

    def _saveObjectsToDisk(self):
        def save_object(obj, filename):
            with open(filename, 'wb') as output:
                pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

        save_object(self.exported_pipeline, 'exportedPipeline.pkl')
        save_object(self.neatData, 'NeatData.pkl')

    def _createTrainedModelPipelineFile(self):
        with open('trainedmodelpipeline.py', 'w') as fileOut:
            fileOut.write("""

import pandas as pd
import pickle

class TrainedModelPipeline:

    def __init__(self):
        self.exportedPipeline = None
        self.neatData = None
        self.testX = None
        self.cleanTestX = None
        self.results = None
        self.resultsDf = None

    def execute(self):
        self._loadObjects()
        self._getDataset()
        self._cleanDataset()
        self._predict()
        self._concatenatePredictionsToDataframe()
        self._saveResultsAsCSV()
        print("Done. Created results.csv")

    def _loadObjects(self):
        with open('exportedPipeline.pkl', 'rb') as input:
            self.exportedPipeline = pickle.load(input)
        with open('NeatData.pkl', 'rb') as input:
            self.neatData = pickle.load(input)

    def _getDataset(self):
        self.testX = pd.read_csv('test_iris.csv') # Edit: Your dataset

    def _cleanDataset(self):
        self.cleanTestX = self.neatData.cleanTestDataset(self.testX)

    def _predict(self):
        self.results = self.exportedPipeline.predict(self.cleanTestX)
        self.results = self.neatData.convertYToStringsOrNumbersForPresentation(self.results)

    def _concatenatePredictionsToDataframe(self):
        self.resultsDf = pd.DataFrame(self.results)
        self.resultsDf = pd.concat([self.testX, self.resultsDf], axis=1)

    def _saveResultsAsCSV(self):
        self.resultsDf.to_csv('./results.csv')

trainedModelPipeline = TrainedModelPipeline()
trainedModelPipeline.execute()
""")

modelPipeline = ModelPipeline()
modelPipeline.execute()
