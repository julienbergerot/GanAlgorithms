from models.models import GAN,Pix2pix,CycleGAN,Classifier,ConditionalGan

def createModel(opt,dataset) :
    if opt["name"] == "Gan" : 
        model = GAN(dataset,opt["dir"])
        print("Model {} created.".format(opt["name"]))
        return model

    if opt["name"] == "pix2pix" : # we also need the target outputs
        model = Pix2pix(dataset["Inputs"],dataset["Ouputs"],opt["dir"])
        print("Model {} created.".format(opt["name"]))
        return model

    if opt["name"] == "cycleGan" : # we need to load two different folders
        model = CycleGAN(opt["dir"],dataset["ImagesA"],dataset["ImagesB"],inShape=dataset["ImagesA"][0].shape)
        print("Model {} created.".format(opt["name"]))
        return model

    if opt["name"] == "classifier" :
        model = Classifier(dataset["Inputs"],dataset["Labels"],dataset["NumClasses"],opt["dir"],dataset["Classes"])
        print("Model {} created.".format(opt["name"]))
        return model

    if opt["name"] == "conditionalGan" :
        model = ConditionalGan(dataset["Inputs"],dataset["Labels"],dataset["NumClasses"],opt["dir"],dataset["Classes"])
        print("Model {} created.".format(opt["name"]))
        return model

def trainModel(opt,model) :
    if opt["name"] == "Gan" : 
        print("Training the model")
        model.train(n_epochs=200,n_batch=128)

    if opt["name"] == "pix2pix" :
        print("Training the model")
        model.train(n_epochs=10, n_batch=1)

    if opt["name"] == "cycleGan" : 
        print("Training the model")
        model.train(n_epochs=100, n_batch=1)

    if opt["name"] == "classifier" :
        print("Training the model")
        model.train() 

    if opt["name"] == "conditionalGan" :
        print("Training the model")
        model.train()  

def useModel(opt,model) :
    if opt["name"] == "Gan" : 
        model.createImages()

    if opt["name"] == "pix2pix" :
        model.predict()

    if opt["name"] == "cycleGan" : 
        model.transform()

    if opt["name"] == "classifier" :
        model.predict()  

    if opt["name"] == "conditionalGan" :
        model.predict()  