from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import  LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datasetLoader import simpledatasetloader
from preprocessing import simplepreprocessor
from imutils import paths
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="Path to Dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
	help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1, help="# of jobs for KNN algorithm (-1 uses all cores)")
args = vars(ap.parse_args())
# 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

print("[INFO] LOADING IMAGES")
imagePaths = list(paths.list_images(args["dataset"]))

#Run proceprocessing pipelines on dataset
sp = simplepreprocessor.SimplePreprocessor(width=32, height=32)
dataloader = simpledatasetloader.SimpleDataSetLoader([sp])
(data, labels) = dataloader.load(imagePaths, verbose=500)
# reshape it to single dimension vector
print("data shape", data.shape)
data = data.reshape((data.shape[0],  32*32*3))
print("[INFO] MEMORY of IMAGE DATA MATRIX {:.1f}MB".format(data.nbytes / (1024*1024.0)))


le = LabelEncoder()
labels = le.fit_transform(labels)
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, random_state=42)

print("[INFO] evaluting KNN classifier ")
model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX), target_names=le.classes_))

# HOW TO RUN THIS SCRIPT : python knn.py --dataset ../dataset/animals


