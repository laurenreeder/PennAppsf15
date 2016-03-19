from sklearn.svm import SVC
from generate_features import featureVectorsFromDirectory as fv

def train_classifier(X, y):
	clf = SVC()
	clf.fit(X, y)
	return clf

def train_with_images(test_dirs, labels):
	if len(labels) != len(test_dirs):
		raise ValueError("Different number of image categories than labels for classification")
	# Get feature vectors for each directory 
 	X = [fv(test_dir) for test_dir in test_dirs]
 	y = []
 	# Add a label to label array for each image
 	for i, label in enumerate(labels):
 		y += [str(label)] * len(X[i])
 	# get all the feature vectors into a single array for the classifier
 	X = [x for feature_list in X for x in feature_list]
 	clf = train_classifier(X, y)
 	return clf

def test():
	svc = train_with_images(["/Users/ella/Documents/cis400/101_ObjectCategories/bonsai", 
					"/Users/ella/Documents/cis400/101_ObjectCategories/crocodile_head"],
					 ["bonsai", "croc"])
	X_test = fv("/Users/ella/Documents/cis400/101_ObjectCategories/crocodile_head")
	print svc.predict(X_test)

test()