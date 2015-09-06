from pyspark.mllib.classification import LogisticRegressionWithLBFGS, NaiveBayes, SVMWithSGD, \
    LogisticRegressionWithSGD

import pyspark.mllib.classification
from pyspark.mllib.tree import DecisionTree, GradientBoostedTrees
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.context import SparkContext
from pyspark.mllib.clustering import GaussianMixture
import numpy as np
from sklearn import metrics, datasets
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import sys, math, random
from functools import partial

multinomial_classifiers = {
    'NaiveBayes': NaiveBayes.train,
    'LogisticRegressionWithLBFGS': LogisticRegressionWithLBFGS.train,
    'DecisionTree': DecisionTree.trainClassifier,
}

binomial_classifiers = {
    'NaiveBayes': NaiveBayes.train,
    'SVMWithSGD': SVMWithSGD.train,
    'LogisticRegressionWithLBFGS': LogisticRegressionWithLBFGS.train,
    'LogisticRegressionWithSGD': LogisticRegressionWithSGD.train,
    'DecisionTree': DecisionTree.trainClassifier,
    'GradientBoostedTrees': GradientBoostedTrees.trainClassifier
}

with_numClasses = {
    'LogisticRegressionWithLBFGS',
    'DecisionTree',
}

with_categoricalFeatures = {
    'DecisionTree',
    'GradientBoostedTrees'
}

with_regLambda = {
    'LogisticRegressionWithLBFGS',
    'LogisticRegressionWithSGD',
}
reg_lambda = 0.000001

def train_classifier(data, classifier, train_kwargs={}):

    # Split data aproximately into training (60%) and test (40%)
    training, test = data.randomSplit([0.6, 0.4], seed = 0)

    # Train a classifier
    model = classifier(training, **train_kwargs)

    # Make prediction and test accuracy.
    predictions = model.predict(test.map(lambda p : p.features))
    results = predictions.zip(test.map(lambda p : p.label))

    accuracy = 1.0 * results.filter(lambda (x, v): x == v).count() / test.count()

    # print "bayes results... accuracy: {}".format(accuracy)
    return accuracy

def train_classifier_on_file(file_parser, filepath, classifier_name, num_classes=None):

    sc = SparkContext("local", classifier_name)

    data, num_classes = get_classified_data(sc, file_parser, filepath, num_classes)

    train_classifier_on_data(data, num_classes, classifier_name)

    sc.stop()

def train_classifier_on_data(data, num_classes, classifier_name):
    train_kwargs = {}
    if classifier_name in with_numClasses:
        train_kwargs['numClasses'] = num_classes

    if classifier_name in with_categoricalFeatures:
        train_kwargs['categoricalFeaturesInfo'] = {}

    if classifier_name in with_regLambda:
        train_kwargs['regParam'] = reg_lambda

    if num_classes > 2:
        if classifier_name not in multinomial_classifiers:
            raise ValueError('classifier_name is invalid. Options are %s'
                                % multinomial_classifiers.keys())

        return train_classifier(data, multinomial_classifiers[classifier_name],
                            train_kwargs=train_kwargs)

    elif num_classes == 2:
        if classifier_name not in binomial_classifiers:
            raise ValueError('classifier_name is invalid. Options are %s'
                                % binomial_classifiers.keys())

        return train_classifier(data, binomial_classifiers[classifier_name],
                            train_kwargs=train_kwargs)


def get_classified_data(sc, file_parser, filepath, num_classes=None):
    data = file_parser(sc.textFile(filepath))
    if num_classes is None:
        labels = data.groupBy(lambda x: x.label).collect()
        num_classes = len(labels)
    if num_classes < 2:
        raise ValueError('Number of classes in dataset must be greater than 1')
    return data, num_classes


def get_classifier_results(file_parser, filepath, num_classes=None):
    sc = SparkContext("local", "Classifier Performance")
    data, num_classes = get_classified_data(sc, file_parser, filepath, num_classes)
    classifiers = binomial_classifiers if num_classes == 2 else multinomial_classifiers
    accuracies = map(lambda class_name: (class_name, train_classifier_on_data(data, num_classes,
                                                                              class_name)),
                     classifiers.keys())
    sc.stop()
    print accuracies


# Gaussian mixture - clusters data, but must pass in desired # of clusters
def gaussian(filepath, n):
    sc = SparkContext("local", "Gaussian App")
    data = sc.textFile(filepath)
    parsedData = data.map(lambda line: np.array([float(x) for x in line.strip().split(' ')]))

    # Build the model (cluster the data)
    gmm = GaussianMixture.train(parsedData, n)

    # output parameters of model
    for i in range(2):
        print ("weight = ", gmm.weights[i], "mu = ", gmm.gaussians[i].mu,
                            "sigma = ", gmm.gaussians[i].sigma.toArray())

    # Metric for unsupervised - silhouette coefficient, higher numbers ~ better fit
    labels = gmm.predict(parsedData).collect()
    features = np.array(map(lambda x : x.strip().split(" "), data.collect()))
    features = features.reshape(features.size/2, 2)
    score = metrics.silhouette_score(features, np.array(labels), metric='euclidean')
    print "score: {}".format(score)

def parse_labeled_vectors(sc_file):
    def parseLine(line):
        parts = line.split(',')
        label = float(parts[0])
        features = Vectors.dense([float(x) for x in parts[1].split(' ')])
        return LabeledPoint(label, features)
    return sc_file.map(parseLine)

def parse_csv(sc_file):
    def parseLine(line):
        parts = line.split(',')
        label = float(parts[-1])
        features = Vectors.dense([float(x) for x in parts[:-1]])
        return LabeledPoint(label, features)
    return sc_file.map(parseLine)

train_classifier_on_labeled_vectors = partial(train_classifier_on_file, parse_labeled_vectors)
results_on_labeled_vectors = partial(get_classifier_results, parse_labeled_vectors)

def main(argv):
  # Check for desired input later... look at length of argv for various models
#    gaussian(argv[1],int(argv[2]))
    print sys.argv
    if sys.argv[1] == 'classifier':
        if len(sys.argv) == 5:
            train_classifier_on_labeled_vectors(str(sys.argv[3]),
                                                str(sys.argv[2]),
                                                int(sys.argv[4]))
        elif len(sys.argv) == 4:
            train_classifier_on_labeled_vectors(str(sys.argv[3]),
                                                str(sys.argv[2]))
        elif len(sys.argv) == 3:
            get_classifier_results(parse_csv, str(sys.argv[2]))
if __name__ == '__main__':
    main(sys.argv)


