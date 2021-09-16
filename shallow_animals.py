# usage  python3 shallow_animals.py --dataset dataset/animals/

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing import imagetoarraypreprocessor
from preprocessing import simplepreprocessor
from datasetLoader import SimpleDataSetLoader
from nn.conv import ShallowNet
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to dataset")
args = vars(ap.parse_args())

print("[INFO] LOADING IMAGES")
imagePaths = paths.list_images(args["dataset"])

# initialize the image processor
# resize to (32,32)
sp = simplepreprocessor.SimplePreprocessor(32, 32)
iap = imagetoarraypreprocessor.ImageToArrayPreprocessor()

dataLoader = SimpleDataSetLoader([sp, iap])
(data, labels) = dataLoader.load(imagePaths, verbose=-1)
# normalize the image
data = data.astype('float') / 255.0

# partition data into train test split
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, train_size=0.75, random_state=1)

# convert labels into one hot encoder vector
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("[INFO]  compiling model")
opt = SGD(lr=0.005)
model = ShallowNet().build(width=32, height=32, depth=3, classes=3)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])

print("[INFO] Training Network")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbose=1)

print('[INFO] Evaluating network')
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=['cat', 'dog', 'panda']))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
