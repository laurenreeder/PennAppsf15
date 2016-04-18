import random
from ml import svm, generate_features
from sys import argv


class_1 = generate_features.featureVectorsFromDirectory(argv[1])
class_2 = generate_features.featureVectorsFromDirectory(argv[2])


all_features = [(feature, 'class_1') for feature in class_1] + [(feature, 'class_2') for feature in class_2]
random.shuffle(all_features)


features = all_features[:5]
curr_features = all_features[5:]
curr_active_features = all_features[:5]
active_features = all_features[5:]
curr_model = svm.train_classifier([x[0] for x in all_features[5:]], [x[1] for x in all_features[5:]])
active_model = curr_model

def train(features):
    return svm.train_classifier([x[0] for x in features], [x[1] for x in features])

def sorted_features(clf, features):
    vals = clf.decision_function([x[0] for x in features])
    dec_vals = map(lambda dists: sum(map(abs, dists)), vals)
    return [x for (y,x) in sorted(zip(dec_vals, features))]


def get_accuracy(clf, features):
    preds = clf.predict([x[0] for x in features])
    actual = [x[1] for x in features]
    return len([x for (x,y) in zip(preds, actual) if x == y])/float(len(features))

for i in xrange(1, 10):
    active_features = sorted_features(active_model, active_features)
    curr_active_features += active_features[:5]
    active_features = active_features[5:]
    curr_features += features[:5]
    features = features[5:]
    active_model = train(curr_active_features)
    curr_model = train(curr_features)
    print "active", get_accuracy(active_model, all_features)
    print "regular", get_accuracy(curr_model, all_features)




