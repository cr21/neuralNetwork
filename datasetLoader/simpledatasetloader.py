import numpy as np
import cv2
import os

class SimpleDataSetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors= preprocessors
        if preprocessors is None:
            self.preprocessors = []

    def load(self, imagePath, verbose=-1):
        data=[]
        labels=[]
        # iterateover every path and then apply and run preprocessing pipelines and return
        for idx, _imagePath in enumerate(imagePath):
            # read the image
            image = cv2.imread(_imagePath)
            # /path/to/dataset/{class}/{image}.jpg
            label = _imagePath.split(os.path.sep)[-2]

            if self.preprocessors is not  None:
                for p in self.preprocessors:
                    image = p.preprocessImage(image)

            data.append(image)
            labels.append(label)

            if verbose > 0 and idx > 0 and (idx + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(idx + 1, len(_imagePath)))

        return (np.array(data), np.array(labels))
