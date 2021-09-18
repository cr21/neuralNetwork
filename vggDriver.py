from nn.conv import vgg
from sklearn.preprocessing import  LabelBinarizer
from sklearn.metrics import  classification_report
from tensorflow.keras.optimizers import  SGD
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import  backend as K
import  numpy as np
import matplotlib.pyplot as plt
import  argparse

ap = argparse.ArgumentParser()
ap.add_argument('-o','--output', required=True, help='path to putput loss and accuracy')
args = ap.parse_args()

print("[Info] Loading CIFAR 10 dataset")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype('float')/255.0;
testX = testX.astype('float')/255.0;

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

labelNames = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

print("compile the model")
opt = SGD(lr=0.01, decay=0.01/40, momentum=0.9, nesterov=True)
model = vgg.MiniVGG.build(height=32, width=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy",optimizer=opt, metrics=['accuracy'])

print("[INFO] Training network")
EPOCHS=10
his = model.fit(trainX,trainY, validation_data=(testX,testY), batch_size=64, epochs=EPOCHS,verbose=1)


print("[info] evalution network")
predictions = model.predict(testX, batch_size=64)

print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=[str(x) for x in model.classes_]))
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), his.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), his.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), his.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), his.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args['output'])

print()

# usage python vggDriver.py --output /output/cifar10_minVGG_bn.png
