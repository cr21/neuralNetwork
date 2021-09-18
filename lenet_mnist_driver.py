from nn.conv import Lenet
from sklearn.preprocessing import  LabelBinarizer
from sklearn.metrics import  classification_report
from tensorflow.keras.optimizers import  SGD
from  tensorflow.keras.datasets import mnist
from tensorflow.keras import  backend as K
import  numpy as np
import matplotlib.pyplot as plt

print("[INFO] Loading MNINST data")
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    trainData = trainData.reshape((trainData.shape[0],1,28,28))
    testData = testData.reshape((testData.shape[0], 1, 28, 28))

else:
    trainData = trainData.reshape(( trainData.shape[0], 28,28,1))
    testData = testData.reshape(( testData.shape[0], 28,28,1))

le = LabelBinarizer()
trainLabels = le.fit_transform(trainLabels)
testLabels = le.fit_transform(testLabels)

opt = SGD(lr=0.001)
print("[Info] Build Model")
lenet = Lenet.build(height=28,width=28,depth=1,classes=10)
lenet.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

print("[info] training network")
history = lenet.fit(trainData, trainLabels, validation_data=(testData, testLabels),  batch_size=128,epochs=2, verbose=1)

print("[info] evalution network")
predictions = lenet.predict(testData, batch_size=128)

print(classification_report(testLabels.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=[str(x) for x in le.classes_]))
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 2), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, 2), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 2), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 2), history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()


'''
[info] evalution network
              precision    recall  f1-score   support

           0       0.98      0.98      0.98       980
           1       0.98      0.99      0.98      1135
           2       0.96      0.97      0.96      1032
           3       0.96      0.97      0.97      1010
           4       0.98      0.96      0.97       982
           5       0.98      0.96      0.97       892
           6       0.96      0.98      0.97       958
           7       0.96      0.95      0.96      1028
           8       0.92      0.97      0.95       974
           9       0.97      0.94      0.95      1009

    accuracy                           0.97     10000
   macro avg       0.97      0.97      0.97     10000
weighted avg       0.97      0.97      0.97     10000


'''