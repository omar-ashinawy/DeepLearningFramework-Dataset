import numpy as np, idx2numpy
import gzip
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt

class Dataset():
    def __init__(self, rootFilePath):
        self.rootFilePath = rootFilePath
    def __getItem__(self, index):
        raise NotImplementedError
    def __len__(self):
        raise NotImplementedError

class Mnist(Dataset):
    trainingImagesPath = 'train-images-idx3-ubyte.gz'
    trainingLabelsPath = 'train-labels-idx1-ubyte.gz'
    testImagesPath = 't10k-images-idx3-ubyte.gz'
    testLabelsPath = 't10k-labels-idx1-ubyte.gz'
    def __init__(self, rootFilePath, training= True): 
        super(Mnist, self).__init__(rootFilePath)
        self.training = training
        if training:
            imagesPath = self.trainingImagesPath
            labelsPath = self.trainingLabelsPath
        else:
            imagesPath = self.testImagesPath
            labelsPath = self.testLabelsPath
        self.images = self.get_images(os.path.join(self.rootFilePath, imagesPath))
        self.labels = self.get_labels(os.path.join(self.rootFilePath, labelsPath))
        self.__index = 0
    def __getItem__(self, index):
        PILImage = Image.fromarray(self.images[index], mode= 'L') 
        # mode = 'L' indicates 8 bits black and white images
        intLabels = self.labels[index].item()
        return PILImage, intLabels
    def __len__(self):
        return len(self.images)
    def __iter__(self):
        self.__index = 0
        return zip(self.images, self.labels)
    def __next__(self):
        print("next")
        if self.__index <= self.__len__():
            self.__index += 1
            return self.__getItem__(self.__index)
        else:
            raise StopIteration
    def generate(self, index= 0):
        internal_index = index
        while True:
            if internal_index < self.__len__():
                yield self.__getItem__(internal_index)
                internal_index += 1
            else: 
                break
    def showSample(self, startFrom= 0, coloumn= 5, row= 5, figsize= (10, 10)):
        fig = plt.figure(figsize= figsize)
        mnist_iter = self.generate(startFrom)
        for i in range(1, coloumn*row+1):
            img, label = next(mnist_iter)
            ax = fig.add_subplot(row, coloumn, i)
            ax.title.set_text('Label: ' + str(label))
            plt.imshow(img)
            plt.axis('off')
        plt.show()
    def read_idx_file(self, f, num_bytes= 4, endianness= 'big'):
        return int.from_bytes(f.read(num_bytes), endianness)
    def get_images(self, compressedFilePath):
        with gzip.open(compressedFilePath, 'r') as f:
            magicNumber = self.read_idx_file(f) # Used to make sure that the file is read correctly.
            try:
                assert magicNumber == 2051 
            except:
                print('Error getting images (Magic number error)')
                raise SystemExit()
            num_images = self.read_idx_file(f) 
            num_rows = self.read_idx_file(f)
            num_coloumns = self.read_idx_file(f)
            rest_values = f.read()
            images = np.frombuffer(rest_values, dtype = np.uint8).reshape((num_images, num_rows, num_coloumns))
            return images
    def get_labels(self, compressedFilePath):
        with gzip.open(compressedFilePath, 'r') as f:
            magicNumber = self.read_idx_file(f) # Used to make sure that the file is read correctly.
            try:
                assert magicNumber == 2049 
            except:
                print('Error in training labels (Magic number error)')
                raise SystemExit()
            num_labels = self.read_idx_file(f)
            rest_values = f.read()
            labels = np.frombuffer(rest_values, dtype = np.uint8).reshape((num_labels, 1))
            return labels

mnist = Mnist('../mnist')
counter = 0
for image, label in mnist.generate():
    counter += 1
print(counter)

# counter = 0
# for image, label in mnist:
#     if counter <= 3:
#         print(label)
#     counter += 1
# x = mnist.generate()
# im, label = next(x)
# iterator = iter(mnist)
# image, label = next(iterator)
# print(label)
# for i in range(0, 60001):
#     im, l = next(iterator)

mnist.showSample(startFrom= 0)