# -*- mode: Python; coding: utf-8 -*-

from classifier import Classifier
from collections import defaultdict

class NaiveBayes(Classifier):
	smoothing_factor = 0.5
    u"""A naïve Bayes classifier."""

    def __init__(self, model={}):
        super(NaiveBayes, self).__init__(model)

    def get_model(self): return self.seen
    def set_model(self, model): self.seen = model
    model = property(get_model, set_model)

    def train(self, instances):
        """Count number of documents containing feature and total count of each label"""
        """labelcount totals instances for each label"""
        labelcount = defaultdict(lambda: 0)
        """feature count totals hits for each feature for each label"""
        featurecount = defaultdict(lambda: defaultdict(lambda: 0))
        instancecount = 0
        for instance in instances:
            for feature in instance.features():
                labelcount[instance.label] += 1
                featurecount[instance.label][feature] += 1
        """from feature counts, compute class conditional probability estimates as log"""
        for label in labelcount.keys():
        	model[label] = {}
        	for feature in featurecount.keys():
        		model[label][feature] = math.log( 
        			(featurecount[feature] + smoothing_factor) / 
        			(labelcount[label] + smoothing_factor*2) )
        """compute priors from label counts"""
        instancecount = sum(labelcount.values())
        for label in labelcount.keys():
        	model[label]['*PRIOR*'] = math.log(labelcount[label] / instancecount)



    def classify(self, instance):
        """Classify an instance using the log probabilities computed during training."""
        for feature in instance.features():
            """sum (log) probabilities in each label and return highest"""

            for label in 
