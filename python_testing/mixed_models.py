import pyspark.mllib.classification
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.context import SparkContext
from pyspark.mllib.clustering import GaussianMixture
from numpy import array
from functools import partial
import sys


def train_classifier(data, classifier, train_kwargs={}):

    # Split data aproximately into training (60%) and test (40%)
    training, test = data.randomSplit([0.6, 0.4], seed = 0)

    # Train a classifier
    model = classifier.train(training, 1.0, **train_kwargs)

    # Make prediction and test accuracy.
    predictionAndLabel = test.map(lambda p : (model.predict(p.features), p.label))
    accuracy = 1.0 * predictionAndLabel.filter(lambda (x, v): x == v).count() / test.count()

    # print "bayes results... accuracy: {}".format(accuracy)
    print accuracy
    return model


def parse_labeled_vectors(sc_file):
    def parseLine(line):
        parts = line.split(',')
        label = float(parts[0])
        features = Vectors.dense([float(x) for x in parts[1].split(' ')])
        return LabeledPoint(label, features)
    return sc_file.map(parseLine)


def train_on_file(file_parser, filepath, classifier_name, num_classes=2):
    print "NAME:", classifier_name
    if hasattr(pyspark.mllib.classification, classifier_name):
        sc = SparkContext("local", "Bayes App")
        data = file_parser(sc.textFile(filepath))
        train_classifier(data, getattr(pyspark.mllib.classification, classifier_name),
                         train_kwargs={'numClasses':num_classes})
        sc.stop()

train_classifier_on_file = partial(train_on_file, parse_labeled_vectors)

# Gaussian mixture - clusters data, but must pass in desired # of clusters
def gaussian(filepath, n):
    sc = SparkContext("local", "Gaussian App")
    data = sc.textFile(filepath)
    parsedData = data.map(lambda line: array([float(x) for x in line.strip().split(' ')]))

    # Build the model (cluster the data)
    gmm = GaussianMixture.train(parsedData, n)

    # output parameters of model
    for i in range(2):
        print ("weight = ", gmm.weights[i], "mu = ", gmm.gaussians[i].mu,
                            "sigma = ", gmm.gaussians[i].sigma.toArray())


def main(argv):
  # Check for desired input later... look at length of argv for various models
    #gaussian(argv[1],int(argv[2]))
    if sys.argv[1] == 'classifier':
        train_classifier_on_file(str(sys.argv[3]), str(sys.argv[2]), int(sys.argv[4]))


if __name__ == '__main__':
    main(sys.argv)


