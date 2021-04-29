import tensorflow as tf
from models.modelHelpers import decoder_block,define_encoder_block,resnet_block
from keras.models import Model
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from keras.datasets.cifar10 import load_data

from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model,Sequential,Input
from math import log,sqrt
import tensorflow as tf

from keras.layers import LeakyReLU,Dense,Reshape,BatchNormalization,Dropout,Concatenate,Activation,LeakyReLU,Conv2DTranspose,Conv2D,Flatten,Embedding
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

# GAN creates images based on real ones // It tries to mimic reality but its only loss is due to its discriminator
# this model doesn't require any targets
class GAN : 
    def __init__(self,dataset : np.array,path : str,latentDim = 100,opt={}) -> None :
        self.inShape = dataset[0].shape
        self.latentDim = latentDim
        self.dataset = dataset
        self.defineDiscriminator(self.inShape)
        self.defineGenerator(latentDim,self.inShape)
        self.defineGan(self.generator,self.discriminator)
        self.path = os.path.join(path,"__model__")
        if not os.path.isdir(self.path) :
            os.mkdir(self.path)
        self.pathSave = os.path.join(path,"outputs")
        if not os.path.isdir(self.pathSave) :
            os.mkdir(self.pathSave)

    def load(self) -> None :
        files = os.listdir(self.path)
        assert len(files) > 0 , "There are no models to load in this directory, {}.".format(self.path)
        oldest = files[0]
        for file in files[1:] : 
            if int(file.split("generator_model_")[1].split(".h5")[0]) > int(oldest.split("generator_model_")[1].split(".h5")[0]) :
                oldest = file
        self.generator = load_model(os.path.join(self.path,oldest))
        print("Model successfully loaded.")

    def generate_real_samples(self, n_samples : int) -> tuple:
        # choose random instances
        ix = np.random.randint(0, self.dataset.shape[0],n_samples)
        # retrieve selected images
        X = self.dataset[ix]
        # generate 'real' class labels (1)
        y = np.ones((n_samples, 1))
        sampl = np.random.uniform(low=-0.1, high=0, size=(n_samples,1))
        return X, y #+ sampl
        
    def generate_latent_points(self, n_samples : int) -> np.array:
        # generate points in the latent space
        x_input = np.random.randn(self.latentDim * n_samples)
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(n_samples, self.latentDim)
        return x_input

    def generate_fake_samples(self, n_samples : int) -> tuple:
        # generate points in latent space
        x_input = self.generate_latent_points(n_samples)
        # predict outputs
        X = self.generator.predict(x_input)
        # create 'fake' class labels (0)
        y = np.zeros((n_samples, 1))
        sampl = np.random.uniform(low=0.0, high=0.1, size=(n_samples,1))
        return X, y # + sampl

    def save_plot(self,examples : np.array, epoch, n=3) -> None:
        # scale from [-1,1] to [0,1]
        examples = (examples + 1) / 2.0
        # plot images
        for i in range(n * n):
            # define subplot
            plt.subplot(n, n, 1 + i)
            # turn off axis
            plt.axis('off')
            # plot raw pixel data
            plt.imshow(examples[i])
        # save plot to file
        filename = 'generated_plot_e%03d.png' % (epoch+1)
        plt.savefig(os.path.join(self.pathSave,filename))
        plt.close()

    def summarize_performance(self,epoch : int, n_samples=9) -> None:
        # prepare real samples
        X_real, y_real = self.generate_real_samples(n_samples)
        # evaluate discriminator on real examples
        _, acc_real = self.discriminator.evaluate(X_real, y_real, verbose=0)
        # prepare fake examples
        x_fake, y_fake = self.generate_fake_samples(n_samples)
        # evaluate discriminator on fake examples
        _, acc_fake = self.discriminator.evaluate(x_fake, y_fake, verbose=0)
        # summarize discriminator performance
        print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
        # save plot
        self.save_plot(x_fake, epoch)
        # save the generator model tile file
        filename = 'generator_model_%03d.h5' % (epoch+1)
        self.generator.save(os.path.join(self.path,filename))

    def train(self, n_epochs=200, n_batch=128) -> None:
        bat_per_epo = int(self.dataset.shape[0] / n_batch)
        half_batch = int(n_batch / 2)
        # manually enumerate epochs
        for i in range(n_epochs):
            # enumerate batches over the training set
            for j in range(bat_per_epo):
                # get randomly selected 'real' samples
                X_real, y_real = self.generate_real_samples(half_batch)
                # update discriminator model weights
                d_loss1, _ = self.discriminator.train_on_batch(X_real, y_real)
                # generate 'fake' examples
                X_fake, y_fake = self.generate_fake_samples(half_batch)
                # update discriminator model weights
                d_loss2, _ = self.discriminator.train_on_batch(X_fake, y_fake)
                # prepare points in latent space as input for the generator
                X_gan = self.generate_latent_points(n_batch)
                # create inverted labels for the fake samples
                y_gan = np.ones((n_batch, 1))
                # update the generator via the discriminator's error
                g_loss = self.gan.train_on_batch(X_gan, y_gan)
                # summarize loss on this batch
                print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                    (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
            # evaluate the model performance, sometimes
            if i % 1 == 0:
                self.summarize_performance(i)

    def createImages(self) -> None :
        latentPoints = self.generate_latent_points(9)
        X = self.generator.predict(latentPoints)
        self.save_plot(X)

    def defineDiscriminator(self,image_shape : tuple) -> None :
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # source image input
        in_src_image = Input(shape=image_shape)
        # concatenate images channel-wise
        merged = in_src_image
        # C64
        d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
        d = LeakyReLU(alpha=0.2)(d)
        # C128
        d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        # d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C256
        d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        # d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C512
        d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        # d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # second last output layer
        d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
        # d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # patch output
        d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
        # to predict a single probability
        d = Flatten()(d)
        d = Dropout(0.4)(d)
        patch_out = Dense(1,activation="sigmoid")(d)
        
        # define model
        model = Model(in_src_image, patch_out)
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt,metrics="accuracy")
        self.discriminator = model

    def defineGenerator(self,latentDim : int,inShape : tuple) -> None :
        model = Sequential()
        init = RandomNormal(mean=0.0, stddev=0.02)
        # foundation for 4x4 image
        n_nodes = 256 * 4 * 4
        model.add(Dense(n_nodes, input_dim=latentDim))
        # model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((4, 4, 256)))

        nbLayers = int(log(inShape[0] // 4,2))
        for _ in range(nbLayers) :
            # upsample to 8x8
            model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same',kernel_initializer=init))
            # model.add(BatchNormalization())
            model.add(LeakyReLU(alpha=0.2))
    
        model.add(Conv2D(inShape[2], (3,3), activation='tanh', padding='same',kernel_initializer=init))
        self.generator = model

    def defineGan(self,g_model : Model , d_model : Model) -> None:
        # make weights in the discriminator not trainable
        d_model.trainable = False
        # connect them
        model = Sequential()
        # add generator
        model.add(g_model)
        # add the discriminator
        model.add(d_model)
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        self.gan = model

# this model aims at transforming its input into an another image, which is related // the loss is a weighted sum based on its discriminator and the L1 norm
# this model require targets
class Pix2pix :
    def __init__(self,inputs : np.array,targets : np.array,path : str,latentDim = 100,opt={}) -> None :
        self.inputs = inputs
        self.targets = targets
        self.inShape = inputs[0].shape

        self.defineDiscriminatorMerged(targets[0].shape)
        self.defineGenratorPix(image_shape=self.inShape,output_shape=targets[0].shape)
        self.defineGanPix(self.generator,self.discriminator,self.inShape)

        self.pathSave = os.path.join(path,"outputs")
        if not os.path.isdir(self.pathSave) :
            os.mkdir(self.pathSave)
        self.path = os.path.join(path,"__model__")
        if not os.path.isdir(self.path) :
            os.mkdir(self.path)
    
    def load(self) -> None :
        files = os.listdir(self.path)
        assert len(files) > 0 , "There are no models to load in this directory, {}.".format(self.path)
        oldest = files[0]
        for file in files[1:] : 
            if int(file.split("model_")[1].split(".h5")[0]) > int(oldest.split("model_")[1].split(".h5")[0]) :
                oldest = file
        self.generator = load_model(os.path.join(self.path,oldest))
        print("Model successfully loaded.")

    def generate_real_samples(self,n_samples : int, patch_shape : int) -> tuple:
        # unpack dataset
        trainA, trainB = self.inputs,self.targets
        # choose random instances
        ix = np.random.randint(0, trainA.shape[0], n_samples)
        # retrieve selected images
        X1, X2 = trainA[ix], trainB[ix]
        # generate 'real' class labels (1)
        y = np.ones((n_samples, patch_shape, patch_shape, 1))
        sampl = np.random.uniform(low=-0.1, high=0.1, size=(n_samples, patch_shape, patch_shape, 1))
        return [X1, X2], y + sampl

    def generate_fake_samples(self,samples : np.array, patch_shape : int) -> tuple:
        # generate fake instance
        X = self.generator.predict(samples)
        # create 'fake' class labels (0)
        y = np.zeros((len(X), patch_shape, patch_shape, 1))
        sampl = np.random.uniform(low=-0.1, high=0.1, size=(len(X), patch_shape, patch_shape, 1))
        return X, y + sampl

    def summarize_performance(self,step : int,n_samples=3) -> None:
        # select a sample of input images
        [X_realA, X_realB], _ = self.generate_real_samples(n_samples, 1)
        # generate a batch of fake samples
        X_fakeB, _ = self.generate_fake_samples(X_realA, 1)
        # scale all pixels from [-1,1] to [0,1]
        # NORMALISATION
        X_realA = (X_realA + 1) / 2.0
        X_realB = (X_realB + 1) / 2.0
        X_fakeB = (X_fakeB + 1) / 2.0
        # plot real source images
        for i in range(n_samples):
            plt.subplot(3, n_samples, 1 + i)
            plt.axis('off')
            if (X_realA[i].shape) ==2 or X_realA[i].shape[2] == 1 :
                plt.imshow(X_realA[i],cmap='gray')
            else :
                plt.imshow(X_realA[i])
        # plot generated target image
        for i in range(n_samples):
            plt.subplot(3, n_samples, 1 + n_samples + i)
            plt.axis('off')
            if (X_fakeB[i].shape) ==2 or X_fakeB[i].shape[2] == 1 :
                plt.imshow(X_fakeB[i],cmap='gray')
            else :
                plt.imshow(X_fakeB[i])
        # plot real target image
        for i in range(n_samples):
            plt.subplot(3, n_samples, 1 + n_samples*2 + i)
            plt.axis('off')
            if (X_realB[i].shape) ==2 or X_realB[i].shape[2] == 1 :
                plt.imshow(X_realB[i],cmap='gray')
            else :
                plt.imshow(X_realB[i])
        # save plot to file
        filename1 = 'plot_%06d.png' % (step+1)
        plt.savefig(os.path.join(self.pathSave,filename1))
        plt.close()
        # save the generator model
        filename2 = 'model_%06d.h5' % (step+1)
        self.generator.save(os.path.join(self.path,filename2))
        print('>Saved: %s and %s' % (filename1, filename2))

    def train(self, n_epochs=100, n_batch=1) -> None:
        # determine the output square shape of the discriminator
        n_patch = self.discriminator.output_shape[1]
        # unpack dataset
        trainA, trainB = self.inputs,self.targets
        # calculate the number of batches per training epoch
        bat_per_epo = int(len(trainA) / n_batch)
        # calculate the number of training iterations
        n_steps = bat_per_epo * n_epochs
        # manually enumerate epochs
        for i in range(n_steps):
            # select a batch of real samples
            [X_realA, X_realB], y_real = self.generate_real_samples(n_batch, n_patch)
            # generate a batch of fake samples
            X_fakeB, y_fake = self.generate_fake_samples(X_realA, n_patch)
            # update discriminator for real samples
            d_loss1 = self.discriminator.train_on_batch([X_realA, X_realB], y_real)
            # update discriminator for generated samples
            d_loss2 = self.discriminator.train_on_batch([X_realA, X_fakeB], y_fake)
            # update the generator
            g_loss, _, _ = self.gan.train_on_batch(X_realA, [y_real, X_realB])
            # summarize performance
            if i % 10 == 0 :
                print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
            # summarize model performance
            if (i+1) % (bat_per_epo * 10) == 0:
                self.summarize_performance(i)

    def predict(self) :
        predictions = self.generator.predict(self.inputs)
        nb = predictions.shape[0] 
        print("Saving {} triplets of pictures.".format(nb))
        for idx in range(nb) :
            if self.inputs.shape[-1] == 3 :
                inputImage = Image.fromarray(((self.inputs[idx]+1)*127.5).astype(np.uint8))
                fakeImage = Image.fromarray(((predictions[idx]+1)*127.5).astype(np.uint8))
                realImage = Image.fromarray(((self.targets[idx]+1)*127.5).astype(np.uint8)) # sometimes, it'd just be the intput
            else : # gray scale
                inputImage = Image.fromarray((np.squeeze((self.inputs[idx]+1)*127.5)).astype(np.uint8))
                fakeImage = Image.fromarray((np.squeeze((predictions[idx]+1)*127.5)).astype(np.uint8))
                realImage = Image.fromarray((np.squeeze((self.targets[idx]+1)*127.5)).astype(np.uint8)) # sometimes, it'd just be the intput
            inputImage.save(os.path.join(self.pathSave,"{}_0input.png".format(idx)))
            fakeImage.save(os.path.join(self.pathSave,"{}_2fake.png".format(idx)))
            realImage.save(os.path.join(self.pathSave,"{}_1real.png".format(idx)))

    def defineDiscriminatorMerged(self,image_shape : tuple) -> None:
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # source image input
        in_src_image = Input(shape=image_shape)
        # target image input
        in_target_image = Input(shape=image_shape)
        # concatenate images channel-wise
        merged = Concatenate()([in_src_image, in_target_image])
        # C64
        d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
        d = LeakyReLU(alpha=0.2)(d)
        # C128
        d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C256
        d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C512
        d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # second last output layer
        d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # patch output
        d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
        patch_out = Activation('sigmoid')(d)
        # define model
        model = Model([in_src_image, in_target_image], patch_out)
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
        self.discriminator = model

    def defineGenratorPix(self,image_shape=(256,256,3),output_shape=(256,256,3)) -> None:
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # image input
        in_image = Input(shape=image_shape)
        # encoder model
        e1 = define_encoder_block(in_image, 64, batchnorm=False)
        e2 = define_encoder_block(e1, 128)
        e3 = define_encoder_block(e2, 256)
        e4 = define_encoder_block(e3, 512)
        e5 = define_encoder_block(e4, 512)
        e6 = define_encoder_block(e5, 512)
        e7 = define_encoder_block(e6, 512)
        # bottleneck, no batch norm and relu
        b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
        b = Activation('relu')(b)
        # decoder model
        d1 = decoder_block(b, e7, 512)
        d2 = decoder_block(d1, e6, 512)
        d3 = decoder_block(d2, e5, 512)
        d4 = decoder_block(d3, e4, 512, dropout=False)
        d5 = decoder_block(d4, e3, 256, dropout=False)
        d6 = decoder_block(d5, e2, 128, dropout=False)
        d7 = decoder_block(d6, e1, 64, dropout=False)
        # output
        g = Conv2DTranspose(output_shape[2], (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
        out_image = Activation('tanh')(g)
        # define model
        model = Model(in_image, out_image)
        self.generator = model

    def defineGanPix(self,g_model : Model, d_model : Model , image_shape : tuple) -> None:
        # make weights in the discriminator not trainable
        for layer in d_model.layers:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = False
        # define the source image
        in_src = Input(shape=image_shape)
        # connect the source image to the generator input
        gen_out = g_model(in_src)
        # connect the source input and generator output to the discriminator input
        dis_out = d_model([in_src, gen_out])
        # src image as input, generated image and classification output
        model = Model(in_src, [dis_out, gen_out])
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,40])
        self.gan = model

# this model aims at performing the translation of a kind of pictures onto another (for instance all the zebras will become horses)
# this model doesn't require targets, but two differents datasets
class CycleGAN :
    def __init__(self,path : str,datasetA : np.array ,datasetB : np.array ,inShape=(256,256,3)) -> None:
        self.path = os.path.join(path,"__model__")
        if not os.path.isdir(self.path) :
            os.mkdir(self.path)

        self.pathSave = os.path.join(path,"outputs")
        if not os.path.isdir(self.pathSave) :
            os.mkdir(self.pathSave)
        self.dataA = datasetA
        self.dataB = datasetB
        # generate A -> B
        self.generatorA = self.defineGeneratorCycle(datasetA[0].shape)
        # generate B -> A
        self.generatorB = self.defineGeneratorCycle(datasetB[0].shape)
        # A -> Real/Fake
        self.discriminatorA = self.defineDiscriminatorCycle(datasetA[0].shape)
        # B -> Real/Fake
        self.discriminatorB = self.defineDiscriminatorCycle(datasetB[0].shape)
        self.compositeA = self.defineCompositeModel(self.generatorA,self.discriminatorB,self.generatorB,datasetB[0].shape)
        self.compositeB = self.defineCompositeModel(self.generatorB,self.discriminatorA,self.generatorA,datasetB[0].shape)

    def load(self) -> None :
        files = os.listdir(self.path)
        assert len(files) > 0 , "There are no models to load in this directory, {}.".format(self.path)

        filesA = [file for file in files in "g_model_AtoB_" in file]
        filesB = [file for file in files in "g_model_BtoA_" in file]

        oldestA = filesA[0]
        oldestB = filesB[0]
        for file in filesA[1:] : 
            if int(file.split("g_model_AtoB_")[1].split(".h5")[0]) > int(oldestA.split("g_model_AtoB_")[1].split(".h5")[0]) :
                oldestA = file
        for file in filesB[1:] : 
            if int(file.split("g_model_BtoA_")[1].split(".h5")[0]) > int(oldestB.split("g_model_BtoA_")[1].split(".h5")[0]) :
                oldestB = file

        cust = {'InstanceNormalization': InstanceNormalization}
        self.generatorA = load_model(os.path.join(self.path,oldestA),cust)
        print("ModelA successfully loaded.")
        self.generatorB = load_model(os.path.join(self.path,oldestB),cust)
        print("ModelB successfully loaded.")

    def generate_real_samples(self,dataset : np.array, n_samples : int, patch_shape : int) -> tuple:
        # choose random instances
        ix = np.random.randint(0, dataset.shape[0], n_samples)
        # retrieve selected images
        X = dataset[ix]
        # generate 'real' class labels (1)
        y = np.ones((n_samples, patch_shape, patch_shape, 1))
        sampl = np.random.uniform(low=-0.1, high=0.1, size=(n_samples, patch_shape, patch_shape, 1))
        return X, y + sampl

    def generate_fake_samples(self,g_model : Model, dataset : np.array, patch_shape : int) -> tuple:
        # generate fake instance
        X = g_model.predict(dataset)
        # create 'fake' class labels (0)
        y = np.zeros((len(X), patch_shape, patch_shape, 1))
        sampl = np.random.uniform(low=-0.1, high=0.1, size=(n_samples, patch_shape, patch_shape, 1))
        return X, y + sampl

    def save_models(self,step : int, g_model_AtoB : Model, g_model_BtoA : Model) -> None:
        # save the first generator model
        filename1 = 'g_model_AtoB_%06d.h5' % (step+1)
        g_model_AtoB.save(filename1)
        # save the second generator model
        filename2 = 'g_model_BtoA_%06d.h5' % (step+1)
        g_model_BtoA.save(filename2)
        print('>Saved: %s and %s' % (os.path.join(self.path,filename1), os.path.join(self.path,filename2)))

    def summarize_performance(self,step : int, g_model : Model, trainX : np.array, name : str, n_samples=5) -> None:
        # select a sample of input images
        X_in, _ = self.generate_real_samples(trainX, n_samples, 0)
        # generate translated images
        X_out, _ = self.generate_fake_samples(g_model, X_in, 0)
        # scale all pixels from [-1,1] to [0,1]
        X_in = (X_in + 1) / 2.0
        X_out = (X_out + 1) / 2.0
        # plot real images
        for i in range(n_samples):
            plt.subplot(2, n_samples, 1 + i)
            plt.axis('off')
            plt.imshow(X_in[i])
        # plot translated image
        for i in range(n_samples):
            plt.subplot(2, n_samples, 1 + n_samples + i)
            plt.axis('off')
            plt.imshow(X_out[i])
        # save plot to file
        filename1 = '%s_generated_plot_%06d.png' % (name, (step+1))
        plt.savefig(os.path.join(self.pathSave,filename1))
        plt.close()

    def update_image_pool(self,pool : list, images : np.array, max_size=50) -> np.array:
        selected = list()
        for image in images:
            if len(pool) < max_size:
                # stock the pool
                pool.append(image)
                selected.append(image)
            elif np.random.random() < 0.5:
                # use image, but don't add it to the pool
                selected.append(image)
            else:
                # replace an existing image and use replaced image
                ix = np.random.randint(0, len(pool))
                selected.append(pool[ix])
                pool[ix] = image
        return np.asarray(selected)

    def train(self,n_epochs=100, n_batch=1) -> None:
        # determine the output square shape of the discriminator
        n_patch = self.discriminatorA.output_shape[1]
        # prepare image pool for fakes
        poolA, poolB = list(), list()
        # calculate the number of batches per training epoch
        bat_per_epo = int(self.dataA.shape[0] / n_batch)
        # calculate the number of training iterations
        n_steps = bat_per_epo * n_epochs
        # manually enumerate epochs
        for i in range(n_steps):
            # select a batch of real samples
            X_realA, y_realA = self.generate_real_samples(self.dataA, n_batch, n_patch)
            X_realB, y_realB = self.generate_real_samples(self.dataB, n_batch, n_patch)
            # generate a batch of fake samples
            X_fakeA, y_fakeA = self.generate_fake_samples(self.generatorB, X_realB, n_patch)
            X_fakeB, y_fakeB = self.generate_fake_samples(self.generatorA, X_realA, n_patch)
            # update fakes from pool
            X_fakeA = self.update_image_pool(poolA, X_fakeA)
            X_fakeB = self.update_image_pool(poolB, X_fakeB)
            # update generator B->A via adversarial and cycle loss
            g_loss2, _, _, _, _  = self.compositeB.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
            # update discriminator for A -> [real/fake]
            dA_loss1 = self.discriminatorA.train_on_batch(X_realA, y_realA)
            dA_loss2 = self.discriminatorA.train_on_batch(X_fakeA, y_fakeA)
            # update generator A->B via adversarial and cycle loss
            g_loss1, _, _, _, _ = self.compositeA.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
            # update discriminator for B -> [real/fake]
            dB_loss1 = self.discriminatorB.train_on_batch(X_realB, y_realB)
            dB_loss2 = self.discriminatorB.train_on_batch(X_fakeB, y_fakeB)
            # summarize performance
            print('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, dA_loss1,dA_loss2, dB_loss1,dB_loss2, g_loss1,g_loss2))
            # evaluate the model performance every so often
            if (i+1) % (bat_per_epo * 1) == 0:
                # plot A->B translation
                self.summarize_performance(i, self.generatorA, self.dataA, 'AtoB')
                # plot B->A translation
                self.summarize_performance(i, self.generatorB, self.dataB, 'BtoA')
            if (i+1) % (bat_per_epo * 5) == 0:
                # save the models
                self.save_models(i, self.generatorA, self.generatorB)

    def show_plot(self,imagesX, imagesY1, imagesY2):
        images = np.vstack((imagesX, imagesY1, imagesY2))
        titles = ['Real', 'Generated', 'Reconstructed']
        # scale from [-1,1] to [0,1]
        images = (images + 1) / 2.0
        # plot images row by row
        for i in range(len(images)):
            # define subplot
            pyplot.subplot(1, len(images), 1 + i)
            # turn off axis
            pyplot.axis('off')
            # plot raw pixel data
            pyplot.imshow(images[i])
            # title
            pyplot.title(titles[i])
        pyplot.show()

    def predict(self) :
        # A -> B -> A
        A_real = select_sample(self.dataA, 1)
        B_generated  = self.generatorA.predict(A_real)
        A_reconstructed = self.generatorB.predict(B_generated)
        self.show_plot(A_real, B_generated, A_reconstructed)

        # B -> A -> B
        B_real = select_sample(self.dataB, 1)
        A_generated  = self.generatorB.predict(B_real)
        B_reconstructed = self.generatorA.predict(A_generated)
        self.show_plot(B_real, A_generated, B_reconstructed)

    def defineGeneratorCycle(self,image_shape : tuple, n_resnet=9) -> Model:
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # image input
        in_image = Input(shape=image_shape)
        # c7s1-64
        g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)
        g = InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)
        # d128
        g = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)
        # d256
        g = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)
        # R256
        for _ in range(n_resnet):
            g = resnet_block(256, g)
        # u128
        g = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)
        # u64
        g = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)
        # c7s1-3
        g = Conv2D(image_shape[2], (7,7), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization(axis=-1)(g)
        out_image = Activation('tanh')(g)
        # define model
        model = Model(in_image, out_image)
        return model

    def defineCompositeModel(self,g_model_1 : Model, d_model : Model, g_model_2 : Model, image_shape : tuple) -> Model:
        # ensure the model we're updating is trainable
        g_model_1.trainable = True
        # mark discriminator as not trainable
        d_model.trainable = False
        # mark other generator model as not trainable
        g_model_2.trainable = False
        # discriminator element
        input_gen = Input(shape=image_shape)
        gen1_out = g_model_1(input_gen)
        output_d = d_model(gen1_out)
        # identity element
        input_id = Input(shape=image_shape)
        output_id = g_model_1(input_id)
        # forward cycle
        output_f = g_model_2(gen1_out)
        # backward cycle
        gen2_out = g_model_2(input_id)
        output_b = g_model_1(gen2_out)
        # define model graph
        model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
        # define optimization algorithm configuration
        opt = Adam(lr=0.0002, beta_1=0.5)
        # compile model with weighting of least squares loss and L1 loss
        model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)
        return model

    def defineDiscriminatorCycle(self,image_shape : tuple) -> Model:
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # source image input
        in_image = Input(shape=image_shape)
        # C64
        d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
        d = LeakyReLU(alpha=0.2)(d)
        # C128
        d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = InstanceNormalization(axis=-1)(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C256
        d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = InstanceNormalization(axis=-1)(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C512
        d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = InstanceNormalization(axis=-1)(d)
        d = LeakyReLU(alpha=0.2)(d)
        # second last output layer
        d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
        d = InstanceNormalization(axis=-1)(d)
        d = LeakyReLU(alpha=0.2)(d)
        # patch output
        patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
        # define model
        model = Model(in_image, patch_out)
        # compile model
        model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
        return model

# this models classfies the inputs in different classes
# the images' name must contain the name of their class
class Classifier :
    def __init__(self,inputs,targets,numClasses,path,classes) :
        self.inputs = inputs
        self.targets = targets
        self.inShape = inputs[0].shape
        self.createClassifier(inputs[0].shape,numClasses)
        self.count = 0
        self.classes = classes

        self.path = os.path.join(path,"__model__")
        if not os.path.isdir(self.path) :
            os.mkdir(self.path)
        self.pathSave = os.path.join(path,"outputs")
        if not os.path.isdir(self.pathSave) :
            os.mkdir(self.pathSave)

    def train(self,batchSize = 32, epochs = 25) :
        self.classifier.fit(self.inputs,self.targets,batch_size=batchSize,epochs=epochs)
        self.classifier.save(os.path.join(self.path,"model_{}.h5".format(self.count)))
        self.count += 1
    
    def load(self) :
        files = os.listdir(self.path)
        assert len(files) > 0 , "There are no models to load in this directory, {}.".format(self.path)
        oldest = files[0]
        for file in files[1:] : 
            if int(file.split("model_")[1].split(".h5")[0]) > int(oldest.split("generator_model_")[1].split(".h5")[0]) :
                oldest = file
        self.classifier = load_model(os.path.join(self.path,oldest))

    def predict(self) :
        print("Predicting {} images.".format(self.inputs.shape[0]))
        prediction = self.classifier.predict(self.inputs)
        for idx in range(prediction.shape[0]) :
            name = self.classes[np.argmax(prediction[idx])]
            inputImage = Image.fromarray(((self.inputs[idx]+1)*127.5).astype(np.uint8))
            inputImage.save(os.path.join(self.pathSave,"correctLabel_{}__prediction_{}__proba_{}.png".format(self.classes[np.argmax(self.targets[idx])],name,np.amax(prediction[idx]))))

    def createClassifier(self,inputShape : tuple, numClasses : int) -> None :
        init = RandomNormal(stddev=0.02)
        # source image input
        in_src_image = Input(shape=inputShape)
        # concatenate images channel-wise
        merged = in_src_image
        # C64
        d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
        d = LeakyReLU(alpha=0.2)(d)
        # C128
        d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        # d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C256
        d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        # d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C512
        d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        # d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # second last output layer
        d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
        # d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # patch output
        d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
        # to predict a single probability
        d = Flatten()(d)
        d = Dropout(0.4)(d)
        patch_out = Dense(numClasses,activation="softmax")(d)
        
        # define model
        model = Model(in_src_image, patch_out)
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='categorical_crossentropy', optimizer=opt,metrics="accuracy")
        self.classifier = model
# this models creates iamges based on real ones // It is similar to the first GAN but it takes in inputs the label (ie the class) that it should be drawing
# the images' name must contain the name of their class

class ConditionalGan :
    def __init__(self,inputs,targets,numClasses,path,classes,latentDim = 100) : 
        self.inputs = inputs
        self.targets = targets
        self.numClasses = numClasses
        self.classes = classes
        self.latentDim = latentDim

        self.define_discriminator(in_shape=inputs[0].shape)
        # self.define_generator(outputShape=inputs[0].shape)
        self.define_generator(outputShape=inputs[0].shape)
        self.define_gan(self.generator,self.discriminator)

        self.path = os.path.join(path,"__model__")
        if not os.path.isdir(self.path) :
            os.mkdir(self.path)
        self.pathSave = os.path.join(path,"outputs")
        if not os.path.isdir(self.pathSave) :
            os.mkdir(self.pathSave)

    def load(self) :
        files = os.listdir(self.path)
        assert len(files) > 0 , "There are no models to load in this directory, {}.".format(self.path)
        oldest = files[0]
        for file in files[1:] : 
            if int(file.split("generator_model_")[1].split(".h5")[0]) > int(oldest.split("generator_model_")[1].split(".h5")[0]) :
                oldest = file
        self.generator = load_model(os.path.join(self.path,oldest))
        print("Model successfully loaded.")

    def generate_real_samples(self, n_samples : int) -> tuple :
        # split into images and labels
        images, labels = self.inputs,self.targets
        # choose random instances
        ix = randint(0, images.shape[0], n_samples)
        # select images and labels
        X, labels = images[ix], labels[ix]
        # generate class labels
        y = ones((n_samples, 1))
        return [X, labels], y

    def generate_latent_points(self, n_samples : int) -> list :
        # generate points in the latent space
        x_input = np.random.randn(self.latentDim * n_samples)
        # reshape into a batch of inputs for the network
        z_input = x_input.reshape(n_samples, self.latentDim)
        # generate labels
        labels = np.random.randint(0, self.numClasses, n_samples)
        return [z_input, labels]
    
    def generate_fake_samples(self, n_samples : int) -> tuple :
        # generate points in latent space
        z_input, labels_input = self.generate_latent_points(n_samples)
        # predict outputs
        images = self.generator.predict([z_input, labels_input])
        # create class labels
        y = np.zeros((n_samples, 1))
        return [images, labels_input], y

    def define_discriminator(self,in_shape=(28,28,1)) -> None :
        # label input
        in_label = Input(shape=(1,))
        # embedding for categorical input
        li = Embedding(self.numClasses, 50)(in_label)
        # scale up to image dimensions with linear activation
        n_nodes = in_shape[0] * in_shape[1] 
        li = Dense(n_nodes)(li)
        # reshape to additional channel
        li = Reshape((in_shape[0], in_shape[1], 1))(li)
        # image input
        in_image = Input(shape=in_shape)
        # concat label as a channel
        merge = Concatenate()([in_image, li])
        # downsample
        fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(merge)
        fe = LeakyReLU(alpha=0.2)(fe)
        # downsample
        fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        # flatten feature maps
        fe = Flatten()(fe)
        # dropout
        fe = Dropout(0.4)(fe)
        # output
        out_layer = Dense(1, activation='sigmoid')(fe)
        # define model
        model = Model([in_image, in_label], out_layer)
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.discriminator = model

    def define_generator(self,outputShape=(256,256,3)) -> None :
        # label input
        in_label = Input(shape=(1,))
        # embedding for categorical input
        li = Embedding(self.numClasses, 50)(in_label)
        # linear multiplication
        n_nodes = 8 * 8
        li = Dense(n_nodes)(li)
        # reshape to additional channel
        li = Reshape((8, 8, 1))(li)
        # image generator input
        in_lat = Input(shape=(self.latentDim,))
        # foundation for 8x8 image
        n_nodes = 128 * 8 * 8
        gen = Dense(n_nodes)(in_lat)
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = Reshape((8, 8, 128))(gen)
        # merge image gen and label input
        merge = Concatenate()([gen, li])
        gen = merge
        # unsampling to the correctSize
        nbLayers = int(log(int(outputShape[0] // 8),2))
        for i in range(nbLayers) : 
            gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
            gen = LeakyReLU(alpha=0.2)(gen)

        assert gen.shape[1:3] == outputShape[0:2] , "The input shape is required to be a power of 2."
            
        # output
        out_layer = Conv2D(outputShape[2], (7,7), activation='tanh', padding='same')(gen)
        # define model
        model = Model([in_lat, in_label], out_layer)
        self.generator = model

    def define_gan(self,g_model, d_model) -> None:
        # make weights in the discriminator not trainable
        d_model.trainable = False
        # get noise and label inputs from generator model
        gen_noise, gen_label = g_model.input
        # get image output from the generator model
        gen_output = g_model.output
        # connect image output and label input from generator as inputs to discriminator
        gan_output = d_model([gen_output, gen_label])
        # define gan model as taking noise and label and outputting a classification
        model = Model([gen_noise, gen_label], gan_output)
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        self.gan = model

    def save_plot(examples : np.array, labels : np.array) -> None :
        # scale from [-1,1] to [0,1]
        examples = (examples + 1) / 2.0
        # plot images
        for i in range(int(sqrt(examples.shape[0]))**2):
            # define subplot
            plt.subplot(int(sqrt(examples.shape[0])), int(sqrt(examples.shape[0])), 1 + i)
            # turn off axis
            plt.axis('off')
            # plot raw pixel data
            plt.imshow(examples[i])
        # the intended labels
        label = ""
        for idx in range(labels.shape[0]) :
            label += str(self.classes[labels[idx]])
            label += "_"
        # save plot to file
        filename = 'generated_plot_e%03d_%s.png' % (epoch+1,label)
        plt.savefig(os.path.join(self.pathSave,filename))
        plt.close()

    def summarize_performance(self,epoch : int, n_samples=9) -> None:
        # prepare real samples
        [X_real, labels_real], y_real = self.generate_real_samples(n_samples)
        # evaluate discriminator on real examples
        _, acc_real = self.discriminator.evaluate([X_real, labels_real], y_real, verbose=0)
        # prepare fake examples
        [X_fake, labels_fake], y_fake = self.generate_fake_samples(n_samples)
        # evaluate discriminator on fake examples
        _, acc_fake = self.discriminator.evaluate([X_fake, labels_fake], y_fake, verbose=0)
        # summarize discriminator performance
        print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
        # save plot
        self.save_plot(X_fake, labels_fake)
        # save the generator model tile file
        filename = 'generator_model_%03d.h5' % (epoch+1)
        self.genrator.save(os.path.join(self.path,filename))

    def train(self, n_epochs=100, n_batch=128):
        bat_per_epo = int(self.inputs[0].shape[0] / n_batch)
        half_batch = int(n_batch / 2)
        # manually enumerate epochs
        for i in range(n_epochs):
            # enumerate batches over the training set
            for j in range(bat_per_epo):
                # get randomly selected 'real' samples
                [X_real, labels_real], y_real = self.generate_real_samples(half_batch)
                # update discriminator model weights
                d_loss1, _ = self.discriminator.train_on_batch([X_real, labels_real], y_real)
                # generate 'fake' examples
                [X_fake, labels], y_fake = self.generate_fake_samples(half_batch)
                # update discriminator model weights
                d_loss2, _ = self.discriminator.train_on_batch([X_fake, labels], y_fake)
                # prepare points in latent space as input for the generator
                [z_input, labels_input] = self.generate_latent_points(n_batch)
                # create inverted labels for the fake samples
                y_gan = np.ones((n_batch, 1))
                # update the generator via the discriminator's error
                g_loss = self.gan.train_on_batch([z_input, labels_input], y_gan)
                # summarize loss on this batch
                print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                    (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
            if (i+1) % 10 == 0 :
                self.summarize_performance(i)

    def predict(self) :
        # 4 examples per class
        latent_points, labels = self.generate_latent_points(4 * self.numClasses) 
        # create ourselves the labels
        labels = np.asarray([x for _ in range(10) for x in range(self.numClasses)])
        # predict
        predictions = self.generator.predict([latent_points,labels])
        predictions = (predictions + 1 ) / 2

        # save for all the classes
        for classe in range(self.numClasses) :
            results = predictions[4*classe:4*classe + 4] # only for the current class
            for idx in range(4) :
                plt.subplot(2,2,i+1)
                plt.axis("off")
                plt.imshow(results[idx])
            filename = "generated_plot_{}.png".format(self.classes[classe])
            plt.savefig(os.path.join(self.pathSave,filename))
            plt.close()
        