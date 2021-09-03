import cv2

"""
Image PreProcesser helper class
"""
class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocessImage(self, image):
        # Resize the image to fixed width and fixed height
        image = cv2.resize(image, (self.width, self.height), interpolation=self.inter)
        return image
