import random
from ml import svm, generate_features
from sys import argv
import pickle


#class_1 = generate_features.featureVectorsFromDirectory(argv[1])
#class_2 = generate_features.featureVectorsFromDirectory(argv[2])
with open('shirt.pickle') as f:
    class_1 = pickle.load(f)
with open('pants.pickle') as f:
    class_2 = pickle.load(f)
all_features = [(feature, 'class_1') for feature in class_1] + [(feature, 'class_2') for feature in class_2]
random.shuffle(all_features)
train_split = int(0.7 * len(all_features))

train_features = all_features[:train_split]
test_features = all_features[train_split:]

print(len(all_features))


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

increment = train_split / 20

for i in xrange(1, 20):
    training_set = train_features[:(increment*i)]
    clf = train(training_set)
    print "%d training instances: %0.2f accuracy" % (increment*i, get_accuracy(clf, test_features))


init_training = train_features[:increment]
curr_features = init_training
rest_features = train_features[increment:]
curr_model = train(init_training)
for i in xrange(2, 20):
    rest_features = sorted_features(curr_model, rest_features)
    curr_features += rest_features[:increment]
    rest_features = rest_features[increment:]
    curr_model = train(curr_features)
    print "%d active training instances: %0.2f accuracy" % (increment*i, get_accuracy(curr_model, test_features))



# for i in xrange(1, 10):
#     active_features = sorted_features(active_model, active_features)
#     curr_active_features += active_features[:5]
#     active_features = active_features[5:]
#     curr_features += features[:5]
#     features = features[5:]
#     active_model = train(curr_active_features)
#     curr_model = train(curr_features)
#     print "active", get_accuracy(active_model, all_features)
#     print "regular", get_accuracy(curr_model, all_features)




