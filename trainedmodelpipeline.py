

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
        self.testX = pd.read_csv('test.csv', header=None) # Edit: Your dataset

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
