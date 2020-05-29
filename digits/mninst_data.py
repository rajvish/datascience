import numpy as np
import matplotlib.pyplot as plt
import random

class mninst_data:
    def __init__(self,imageFile = "train-images-idx3-ubyte", 
                labelFile = "train-labels-idx1-ubyte"):
        self.imageFile = imageFile
        self.labelFile = labelFile
        self.keyImage = 2051
        self.keyLabel = 2049

        #image Parameters
        self.imgWidth=0
        self.imgHeight=0
        self.nitems=0
        self.initSuccess = True
        if self.openImageFile() == False:
            self.initSuccess = False
        if self.openLabelFile() == False:
            self.initSuccess = False
        self.images = self.getImages()
        self.labels = self.getLabels()
    def openImageFile(self):
        if self.imageFile == None:
            return False
        try:
            self.imHandle = open(self.imageFile,"rb")
        except:
            return False
        key=int.from_bytes(self.imHandle.read(4),byteorder='big')
        if key != self.keyImage: 
            print("Invalid Label  Key",key)
            self.imHandle.close()
            return False
        self.nitems = int.from_bytes(self.imHandle.read(4),byteorder='big')
        self.imgHeight = int.from_bytes(self.imHandle.read(4),byteorder='big')
        self.imgWidth = int.from_bytes(self.imHandle.read(4),byteorder='big')
        #print("number of items = ",self.nitems)
        return True
    def openLabelFile(self):
        if self.labelFile == None:
            self.lblHandle = None
            return False
        try:
            self.lblHandle = open(self.labelFile,"rb")
        except:
            return False
        key=int.from_bytes(self.lblHandle.read(4),byteorder='big')
        if key != self.keyLabel: 
            print("Invalid Label  Key",key)
            return False
        nitems = int.from_bytes(self.lblHandle.read(4),byteorder='big')
        if self.nitems != nitems:
            print("Mismatched number of Items. Images = ", self.nitems,nitems)
            return False
        print("Number of Labels = ",nitems)
        return True
    def getImages(self):
        if self.initSuccess == False:
            print("init Failed")
            return None
        img = np.reshape(list(self.imHandle.read(self.imgHeight*self.imgWidth*self.nitems)),(self.nitems,self.imgWidth*self.imgHeight))
        #images = pd.DataFrame(img,columns=cols)
        return img
    def getLabels(self):
        if self.initSuccess == False:
            print("init Failed")
            return None
        if self.lblHandle == None:
            return None
        labels = [l for l in self.lblHandle.read(self.nitems)]
        #labels =  pd.DataFrame(labels,columns=['label'])
        return np.asarray(labels)
    def getnItems(self):
        if self.initSuccess == False:
            print("init Failed")
            return 0
        return self.nitems
    def getPicwidth(self):
        if self.initSuccess == False:
            print("init Failed")
            return 0,0
        return self.imgWidth,self.imgHeight

    def  getData(self):
        return self.images,self.labels
    def drawImage(self,index):
        plt.imshow(self.images[index:index+1].reshape(28,28),cmap='gray')
        plt.title(self.labels[index])   #Remember Y is a mx1 matrix
        plt.show()
    
def mninst_data_test():
    np.random.seed(0)
    iFile = "train-images-idx3-ubyte"
    lFile = "train-labels-idx1-ubyte"
    data=mninst_data(iFile,lFile)
    nItems=data.getnItems()
    print("Number of Images = ",nItems)
    print("Pixel Width = ",data.getPicwidth())
    idx=random.randrange(0,nItems)
    data.drawImage(idx)
    images,labels = data.getData()
    print("Images  = ",type(images))
    print("labels  = ",type(labels))
    print(images.shape)
    print(labels.shape)

if __name__ == "main":
    mninst_data_test()