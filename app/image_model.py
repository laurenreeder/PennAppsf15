from pyspark.mllib.classification import LogisticRegressionWithLBFGS, NaiveBayes, SVMWithSGD, \
    LogisticRegressionWithSGD

import json
import pyspark.mllib.classification
from pyspark.mllib.tree import DecisionTree, GradientBoostedTrees
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, RidgeRegressionWithSGD, \
    LassoWithSGD, IsotonicRegression
from pyspark.context import SparkContext
from pyspark.mllib.clustering import GaussianMixture
import numpy as np
from sklearn import metrics, datasets
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import sys, math, random
from functools import partial

classifiers = {
    'NaiveBayes': {
        'train_func': NaiveBayes.train,
        'defaults': {
        },
        'type': 'classifier',
        'multinomial': True,
    },
    'LogisticRegressionWithLBFGS': {
        'train_func': LogisticRegressionWithLBFGS.train,
        'defaults': {
        },
        'type': 'classifier',
        'multinomial': True,
    },
    'DecisionTree': {
        'train_func': DecisionTree.trainClassifier,
        'defaults': {
        },
        'type': 'classifier',
        'multinomial': True,
    },

    'SVMWithSGD': {
        'train_func': SVMWithSGD.train,
        'defaults': {
        },
        'type': 'classifier',
        'multinomial': False,
    },
    'LogisticRegressionWithSGD': {
        'train_func': LogisticRegressionWithSGD.train,
        'defaults': {
        },
        'multinomial': False,
    },
    'GradientBoostedTrees': {
        'train_func': GradientBoostedTrees.trainClassifier,
        'defaults': {
        },
        'multinomial': False,
    },
}

regressors = {
    'LinearRegressionWithSGD': {
        'train_func': LinearRegressionWithSGD.train,
        'defaults': {
        },
    },
    'RidgeRegressionWithSGD': {
        'train_func': RidgeRegressionWithSGD.train,
        'defaults': {
        },
    },
    'LassoWithSGD': {
        'train_func': LassoWithSGD.train,
        'defaults': {
        },
    },
    'IsotonicRegression': {
        'train_func': IsotonicRegression.train,
        'defaults': {
        },
    },
    'DecisionTree': {
        'train_func': DecisionTree.trainRegressor,
        'defaults': {
        },
    },
    'GradientBoostedTrees': {
        'train_func': GradientBoostedTrees.trainRegressor,
        'defaults': {
        },
    },
}

models_by_task = {
    'classification': classifiers.keys(),
    'regression': regressors.keys(),
}

with_numClasses = {
    'LogisticRegressionWithLBFGS',
    'DecisionTree',
}

def train_classifier(data, classifier, train_kwargs={}):

    # Split data aproximately into training (60%) and test (40%)
    if split is not None:
        training, test = data.randomSplit(split, seed = 0)

    # Train a classifier
    model = classifier(data, **train_kwargs)

    # Make prediction and test accuracy.
    predictions = model.predict(data.map(lambda p : p.features))
    results = predictions.zip(data.map(lambda p : p.label))

    accuracy = 1.0 * results.filter(lambda (x, v): x == v).count() / data.count()

    return accuracy

def create_model(data, model_trainer, train_kwargs={}, split=None):

    if split is not None:
        training, test = data.randomSplit(split, seed = 0)
    else:
        training, test = data, data

    model = model_trainer(training, **train_kwargs)

    predictions = model.predict(test.map(lambda p : p.features))
    results = predictions.zip(test.map(lambda p : p.label))

    return model, results

def create_and_test_model(data, model_trainer, train_kwargs={}):
    model, results = create_model(data, model_trainer, train_kwargs={})
    if model_type == 'regressor':
        return RegressionMetrics(results).rootMeanSquareError
    elif model_type == 'classifier':
        return 1.0 * results.filter(lambda (x, v): x == v).count() / data.count()


def train_classifier_on_file(file_parser, filepath, classifier_name, num_classes=None):

    sc = SparkContext("local", classifier_name)

    data, num_classes = get_classified_data(sc, file_parser, filepath, num_classes)

    train_classifier_on_data(data, num_classes, classifier_name)

    sc.stop()

def train_classifier_on_data(data, num_classes, classifier_name):
    train_kwargs = {}
    if classifier_name in with_numClasses:
        train_kwargs['numClasses'] = num_classes
    if classifier_name not in classifiers:
        raise ValueError('classifier_name is invalid. Options are %s'
                         % multinomial_classifiers.keys())

    if num_classes > 2 and classifiers[classifier_name]['multinomial']:
        if not classifiers[classifier_name]['multinomial']:
            raise ValueError('This classifier can only process binary output distributions.')

    return train_classifier(data, classifiers[classifier_name],
                            train_kwargs=train_kwargs)


def get_classified_data(sc, file_parser, filepath, num_classes=None):
    data = file_parser(sc.textFile(filepath))
    if num_classes is None:
        labels = data.groupBy(lambda x: x.label).collect()
        num_classes = len(labels)
    if num_classes < 2:
        raise ValueError('Number of classes in dataset must be greater than 1')
    return data, num_classes


def get_classifier_results(file_parser, filepath, num_classes=None, train_args={}):
    sc = SparkContext("local", "Classifier Performance")
    data, num_classes = get_classified_data(sc, file_parser, filepath, num_classes)
    classifiers = binomial_classifiers if num_classes == 2 else multinomial_classifiers
    accuracies = test_all_on_data(data, classifiers, 'classifier', train_args=train_args)
    sc.stop()

    print accuracies

def create_and_test_from_file(file_parser, filepath, model_info, num_classes=None, train_args={}, split=None):
    sc = SparkContext("local", "Regressor Performance")
    data = file_parser(sc.textFile(filepath))

    return create_and_test_model(data, model_info['train_func'], model_info['defaults'],
                                 train_args=train_args, num_classes=num_classes, split=split)

def test_all_on_data(data, model_infos, model_type, train_args={}):
    return map(lambda name, info: (name,
                                   create_and_test_model(data, info['train_func'],
                                                         info['defaults'], model_type,
                                                         train_args=train_args)),
               model_infos.items())

def get_regressor_results(file_parser, filepath, num_classes=None, train_args={}):
    sc = SparkContext("local", "Regressor Performance")
    data = file_parser(sc.textFile(filepath))
    rmses = test_all_on_data(data, regressors, 'regressor', train_args=train_args)
    sc.stop()
    print rmses

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
    return score

def parse_labeled_vectors(sc_file):
    def parseLine(line):
        parts = line.split(',')
        label = float(parts[0])
        features = Vectors.dense([float(x) for x in parts[1].split(' ')])
        return LabeledPoint(label, features)
    return sc_file.map(parseLine)

def parse_json(sc_file):
    return sc_file.map(json.loads)

def parse_seperated(sc_file, sep=','):
    def parseLine(line):
        parts = line.split(sep)
        label = float(parts[-1])
        f_list = [float(x) for x in parts[:-1]]
        non_zero_entries = [(i, f_list[i]) for i in xrange(len(f_list)) if f_list[i] != 0.0]
        features = Vectors.sparse(len(f_list), non_zero_entries)
        return LabeledPoint(label, features)
    return sc_file.map(parseLine)

train_classifier_on_labeled_vectors = partial(train_classifier_on_file, parse_labeled_vectors)
results_on_labeled_vectors = partial(get_classifier_results, parse_labeled_vectors)

def file_extension(filepath):
    return filepath.split('?')[0].split('/')[-1].split('.')[-1]

parser_map = {
    'csv': parse_seperated,
    'tsv': partial(parse_seperated, sep='\t'),
    'json': parse_json,
}

def run_file(filepath, model_name, model_type, file_parser=None, num_classes=None, train_args={}):
    ext = file_extension(filepath)
    if ext in parser_map and file_parser is None:
        file_parser = parser_map[ext]
    if file_parser is None:
        return "file_parser none"

    if model_name == 'all':
        if model_type == 'regression':
            return get_regressor_results(file_parser, filepath, num_classes)
        elif model_type == 'classification':
            return get_classifier_results(file_parser, filepath, num_classes)
    elif model_name in regressors:
        sc = SparkContext("local", model_name)
        data = file_parser(sc.textFile(filepath))
        args = regressors[model_name]['defaults'].copy()
        args.update(train_args)

        return create_and_test_model(data, regressors[model_name]['train_func'],
                                         num_classes=num_classes, train_args=args,
                                         split=[0.6, 0.4])
    elif model_name in classifiers:
        sc = SparkContext("local", model_name)
        data = file_parser(sc.textFile(filepath))
        if num_classes is None:
            labels = data.groupBy(lambda x: x.label).collect()
            num_classes = len(labels)
        args = classifiers[model_name]['defaults'].copy()
        args.update(train_args)
        if model_name in with_numClasses:
            train_args['numClasses'] = num_classes

        return create_and_test_model(data, classifiers[model_name]['train_func'],
                                     train_args=train_args, split=[0.6, 0.4])


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
            get_classifier_results(parse_seperated, str(sys.argv[2]))
if __name__ == '__main__':
    main(sys.argv)


