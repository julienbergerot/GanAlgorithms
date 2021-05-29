import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys

from data.createData import createDataset
from models.createModel import createModel,trainModel,useModel
from utils.show import showOptions

from keras.datasets.cifar10 import load_data

actions = ["test","train"]
models = ["pix2pix","Gan","classifier","conditionalGan","cycleGan"]

# basic options
opt = {}
opt["extension"] = ".png"
opt["size"] = 256 # in priority 2**n
opt["channels"] = 3
opt["crop"] = False
opt["dataAug"] = True
opt["dataMult"] = 30
opt["Crop"] = False
assert len(sys.argv) > 1 , "You did not specify a working directory."
opt["dir"] = sys.argv[1]
opt["name"] = "pix2pix"
opt["mode"] = "8bits"
assert len(sys.argv) > 2, "You did not specify whether you wanted to train or test a model."
opt["action"] = sys.argv[2]
if len(sys.argv) == 4 :       
    assert sys.argv[3] in models , "The model you want to create is not available, available models are {}".format(models)
    opt["name"] = sys.argv[3]

if __name__ == '__main__':

    dataset = createDataset(opt["action"],opt)
    model = createModel(opt,dataset)
    showOptions(opt)

    if opt["action"] == "train" :
        trainModel(opt,model)
    
    if opt["action"] == "test" :
        model.load()
        useModel(opt,model)

    