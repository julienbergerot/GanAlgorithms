from data.loadData import imData

def createDataset(key :str,opt :dict) :
    if opt["name"] == "Gan" : 
        dataset = imData(opt["dir"],opt = opt)
        trainingData = dataset.load(key)
        print("Using {} images for training.".format(trainingData["Inputs"].shape[0]))
        return trainingData["Inputs"]

    if opt["name"] == "pix2pix" : # we also need the target outputs
        dataset = imData(opt["dir"],opt = opt)
        trainingData = dataset.load(key)
        print("Using {} images for training.".format(trainingData["Inputs"].shape[0]))
        return trainingData
    
    if opt["name"] == "cycleGan" : # we need to load two different folders
        dataset = imData(opt["dir"],opt = opt)
        dataset = dataset.loadFolders(key)
        print("Using {} images for training.".format(dataset["ImagesA"].shape[0] + dataset["ImagesB"].shape[0]))
        return dataset

    if opt["name"] == "classifier" :
        dataset = imData(opt["dir"],opt = opt)
        dataset = dataset.loadClasses(key,oneHot=True)
        print("Using {} images for training.".format(dataset["Inputs"].shape[0]))
        return dataset

    if opt["name"] == "conditionalGan" :
        dataset = imData(opt["dir"],opt = opt)
        dataset = dataset.loadClasses(key,oneHot=False)
        print("Using {} images for training.".format(dataset["Inputs"].shape[0]))
        return dataset
