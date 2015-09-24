# -*- mode: Python; coding: utf-8 -*-

from __future__ import division
from classifier import Classifier
from collections import defaultdict
import math

smoothing_factor = .2

"""Used in defaultdict to set the 'miss' probability value"""
"""It was experimentally determined that a greater penalty for misses 
than the standard smoothing produced better results"""
def default_probability():
    return math.log( smoothing_factor/(1000*smoothing_factor) )

class NaiveBayes(Classifier):
    u"""A naïve Bayes classifier."""

    def __init__(self, model={}):
        super(NaiveBayes, self).__init__(model)

    def get_model(self): return self.seen
    def set_model(self, model): self.seen = model
    model = property(get_model, set_model)

    def train(self, instances):
        modeldata = {}
        """Count number of documents containing feature and total count of each label"""

        """labelcount totals instances for each label"""
        labelcount = defaultdict(lambda: 0)

        """feature count totals hits for each feature for each label"""
        """NOTE: All features are considered hits in a dense representation"""
        featurecount = defaultdict(lambda: defaultdict(lambda: 0))
        instancecount = 0
        for instance in instances:
            labelcount[instance.label] += 1
            for feature in instance.features():
                featurecount[instance.label][feature] += 1

        """from feature counts, compute class conditional probability estimates as log"""
        for label in labelcount.keys():
            """instantiate dictionary for this label with default penalty for words not seen"""
            modeldata[label] = defaultdict( default_probability )
            for feature in featurecount[label].keys():
                modeldata[label][feature] = math.log( 
                    (featurecount[label][feature] + smoothing_factor) / 
                    (labelcount[label] + smoothing_factor*2) )

        """compute priors from label counts"""
        instancecount = sum(labelcount.values())
        for label in labelcount.keys():
            modeldata[label]['*PRIOR*'] = math.log(labelcount[label] / instancecount)

        modeldata.pop('', None)
        self.set_model(modeldata)

    def classify(self, instance):
        """Classify an instance using the log probabilities computed during training."""
        labelsum = {}

        """initialize with priors"""
        for label in self.model.keys():
            labelsum[label] = self.model[label]['*PRIOR*']

        for feature in instance.features():
            """sum (log) probabilities in each label and return highest"""
            for label in self.model.keys():
                labelsum[label] += self.model[label][feature]

        """return label with largest sum"""
        return max(labelsum, key=lambda k: labelsum[k])