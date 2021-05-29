import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers

import matplotlib.pyplot as plt
from data.imageUtils import Rotate,Crop,CenterCrop,SaveImg,Norm



class imData :
    def __init__(self,path : str,opt : dict) -> None :
        self.path = path
        self.opt = opt
        self.iRot = Rotate()
        self.iCrop = Crop(opt["size"])
        self.iCenterCrop = CenterCrop(opt["size"])
        self.iSaveImg = SaveImg(os.path.join(os.path.abspath(os.getcwd()),self.path),mode="[-1,1]")
        self.iNorm = Norm(mode="8bits")
    
    # load when inputs and targets are in one directory intputs are A_*.extension and targets B_*.extensions
    def load(self,key : str) -> dict :
        # get the noum of the files
        self.path = os.path.join(self.path,key)
        assert os.path.isdir(os.path.join(os.path.abspath(os.getcwd()),self.path)) , "There are no directory called {} in this path.".format(self.path)

        files = os.listdir(os.path.join(os.path.abspath(os.getcwd()),self.path))
        if "extension" in self.opt.keys() :
            files = [file for file in files if self.opt["extension"] in file]
            assert len(files) > 0 and not (key=="test" and self.opt["name"]=="Gan"), "There are no files with the given extension, {}, in the {} directory.".format(self.opt["extension"],self.path)

        assert len(files) > 0 and not (key=="test" and self.opt["name"]=="Gan"), "There are no files in this directory, {}.".format(self.path)

        As = sorted([file for file in files if "A_" in file])
        Bs = sorted([file for file in files if "B_" in file])

        print("Successfully found {} images.".format(len(As)))

        if len(files) > 1 :
            assert [Image.open(os.path.join(os.path.join(os.path.abspath(os.getcwd()),self.path),file)).size[:2] == Image.open(os.path.join(os.path.join(os.path.abspath(os.getcwd()),self.path),files[0])).size[:2] for file in files[1:]] == [True] * (len(files) - 1 ), "The images are required to have the same shape"

        if len(Bs) != len(As) : # meaning there are no target images ie Bs = []
            Bs = As.copy()

        imagesA = []
        imagesB = []

        for image,target in zip(As,Bs) :
            image = np.array(Image.open((os.path.join(os.path.join(os.path.abspath(os.getcwd()),self.path),image))))
            target = np.array(Image.open((os.path.join(os.path.join(os.path.abspath(os.getcwd()),self.path),target))))
            if len(image.shape) == 2 :
                image = np.reshape(image,(image.shape[0],image.shape[1],1))
            if len(target.shape) == 2 :
                target = np.reshape(target,(target.shape[0],target.shape[1],1))

            # the basic images
            if self.opt["Crop"] :
                im,tar = self.iCenterCrop(image,target)
            else : 
                im,tar = image,target
            imagesA.append(im)
            imagesB.append(tar)

            numberImages = self.opt["dataMult"] if self.opt["dataAug"] and key=="train" else 0
            for _ in range(numberImages) : 
                # 1 image became 30
                im,tar = self.iRot(image,target)
                if self.opt["Crop"] :
                    im,tar = self.iCenterCrop(im,tar)
                else : 
                    im,tar = image,target
                imagesA.append(im)
                imagesB.append(tar)
            
        return {"Inputs" : self.iNorm(np.array(imagesA)) , "Ouputs" :self.iNorm(np.array(imagesB))}

    # load when there are 2 different datasets (horses and zerbas for instance) in two different folders
    def loadFolders(self,key : str) -> dict :
        self.path = os.path.join(self.path,key)
        assert os.path.isdir(os.path.join(os.path.abspath(os.getcwd()),self.path)) , "There are no directory called {} in this path.".format(self.path)

        files = os.listdir(os.path.join(os.path.abspath(os.getcwd()),self.path))
        
        directories = [directory for directory in files if os.path.isdir(os.path.join((os.path.join(os.path.abspath(os.getcwd()),self.path)),directory))]
        assert len(directories) == 2 , "There are not 2 directories in this path, {}.".format(os.path.join(os.path.abspath(os.getcwd()),self.path))

        folderA = directories[0]
        imagesA = []
        filesA = os.listdir(os.path.join((os.path.join(os.path.abspath(os.getcwd()),self.path)),folderA))
        folderB = directories[1]
        imagesB = []
        filesB = os.listdir(os.path.join((os.path.join(os.path.abspath(os.getcwd()),self.path)),folderB))

        # sanity check
        if "extension" in self.opt.keys() :
            files = [file for file in filesA if self.opt["extension"] in file]
            assert len(filesA) > 0 , "There are no files with the given extension, {}, in the {} directory.".format(self.opt["extension"],os.path.join(self.path,folderA))
        if "extension" in self.opt.keys() :
            files = [file for file in filesB if self.opt["extension"] in file]
            assert len(filesB) > 0 , "There are no files with the given extension, {}, in the {} directory.".format(self.opt["extension"],os.path.join(self.path,folderB))

        assert len(filesA) > 0 and len(filesB) > 0, "There are no files in this directory, {}.".format(self.path)

        for image in filesA :
            image = np.array(Image.open((os.path.join(os.path.join(os.path.abspath(os.getcwd()),os.path.join(self.path,folderA)),image))))
            if len(image.shape) == 2 :
                image = image.reshape(image.shape[0],image.shape[1],1)
            # the basic images
            im = self.iCenterCrop(image)
            imagesA.append(im)

            numberImages = self.opt["dataMult"] if self.opt["dataAug"] else 0
            for _ in range(numberImages) : 
                # 1 image became 30
                im = self.iRot(image)
                im = self.iCrop(im)
                imagesA.append(im)
                self.iSaveImg(im,mode="Aucun")

        for image in filesB :
            image = np.array(Image.open((os.path.join(os.path.join(os.path.abspath(os.getcwd()),os.path.join(self.path,folderB)),image))))
            if len(image.shape) == 2 :
                image = image.reshape(image.shape[0],image.shape[1],1)
            # the basic images
            im = self.iCenterCrop(image)
            imagesB.append(im)

            numberImages = self.opt["dataMult"] if self.opt["dataAug"] else 0
            for _ in range(numberImages) : 
                # 1 image became 9
                im = self.iRot(image)
                im = self.iCrop(im)
                imagesB.append(im)
        
        assert imagesB[0].shape == imagesA[0].shape , "The two folders do not contain images of the same shape."

        return {"ImagesA" : self.iNorm(np.array(imagesA)), "ImagesB" : self.iNorm(np.array(imagesB))}

    def loadClasses(self,key : str,oneHot=True) -> dict :
        self.path = os.path.join(self.path,key)
        assert os.path.isdir(os.path.join(os.path.abspath(os.getcwd()),self.path)) , "There are no directory called {} in this path.".format(self.path)

        files = os.listdir(os.path.join(os.path.abspath(os.getcwd()),self.path))
        if "extension" in self.opt.keys() :
            files = [file for file in files if self.opt["extension"] in file]
            assert len(files) > 0 and not(key == "test" and self.opt["name"] == "conditionalGan") , "There are no files with the given extension, {}, in the {} directory.".format(self.opt["extension"],self.path)

        assert len(files) > 0 and not(key == "test" and self.opt["name"] == "conditionalGan"), "There are no files in this directory, {}.".format(self.path)

        if len(files) > 1 :
            assert [Image.open(os.path.join(os.path.join(os.path.abspath(os.getcwd()),self.path),file)).size[:2] == Image.open(os.path.join(os.path.join(os.path.abspath(os.getcwd()),self.path),files[0])).size[:2] for file in files[1:]] == [True] * (len(files) - 1 ), "The images are required to have the same shape"

        print("Successfully found {} images.".format(len(files)))

        classes = []
        for file in files :
            classe = file.split("_")[0]
            if not classe in classes :
                classes.append(classe)
        classes = list(set(classes)) # to be sure

        inputs, labels = [], []
        passs = False
        for file in files : 
            img = np.array(Image.open(os.path.join(os.path.join(os.path.abspath(os.getcwd()),self.path),file)))
            if len(img.shape) == 2 :
                if self.opt["channels"] != 1 :
                    passs = True
                    continue
                else :
                    img = np.reshape(img,(img.shape[0],img.shape[1],1))
            if passs :
                continue
            img = self.iCenterCrop(img)
            inputs.append(img)
            
            if oneHot :
                idx = classes.index(file.split("_")[0])
                label = np.zeros(len(classes))
                label[idx] = 1
            else : 
                label = classes.index(file.split("_")[0])
            labels.append(label)
        
        print("There are {} classes".format(len(classes)))
            
        return {"Inputs" : self.iNorm(np.asarray(inputs)), "Labels" : np.asarray(labels), "NumClasses" : len(classes),"Classes" : classes}
        