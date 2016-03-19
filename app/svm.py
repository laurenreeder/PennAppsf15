import numpy as np
from sklearn import svm

x = np.array(import_data_array)
labels = import_label_array

clf = svm.SVC(kernel="linear", C=1.0)
clf.fit(x,labels)

