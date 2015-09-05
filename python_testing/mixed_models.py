from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.context import SparkContext
from pyspark.mllib.clustering import GaussianMixture
from numpy import array
import sys


def naive(filepath):

    def parseLine(line):
        parts = line.split(',')
        label = float(parts[0])
        features = Vectors.dense([float(x) for x in parts[1].split(' ')])
        return LabeledPoint(label, features)

    sc = SparkContext("local", "Bayes App")
    data = sc.textFile(filepath).map(parseLine)

    # Split data aproximately into training (60%) and test (40%)
    training, test = data.randomSplit([0.6, 0.4], seed = 0)

    # Train a naive Bayes model.
    model = NaiveBayes.train(training, 1.0)

    # Make prediction and test accuracy.
    predictionAndLabel = test.map(lambda p : (model.predict(p.features), p.label))
    accuracy = 1.0 * predictionAndLabel.filter(lambda (x, v): x == v).count() / test.count()

    # print "bayes results... accuracy: {}".format(accuracy)
    


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
    gaussian(argv[1],int(argv[2]))
#  naive(str(sys.argv[1]))  

if __name__ == '__main__':
    main(sys.argv)


