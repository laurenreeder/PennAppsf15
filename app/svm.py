from ml import generate_features as gf
import numpy as np
from sklearn import svm

def train_classifier(X, y):
	clf = svm.SVC(kernel="linear")
	clf.fit(X, y)
	return clf
	
def test_SVC(train_dir1, train_dir2, test_dir):
	feat_1 = gf.featureVectorsFromDirectory(train_dir1)
	feat_2 = gf.featureVectorsFromDirectory(train_dir2)
	lab_1 = [1] * len(feat_1)
	lab_2 = [0] * len(feat_2)

	X = np.concatenate((feat_1, feat_2))
	y = lab_1 + lab_2

	clf = train_classifier(X, y)
	test_predictions = clf.predict(gf.featureVectorsFromDirectory(test_dir)) 
	print test_predictions

brains = '/Users/ella/Documents/cis400/101_ObjectCategories/brain/'
dolphins = '/Users/ella/Documents/cis400/101_ObjectCategories/dolphin/'
test_SVC(brains, dolphins, '/Users/ella/Documents/cis400/Senior-Design/test_images/')