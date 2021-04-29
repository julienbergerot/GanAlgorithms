# Algotithms for image analysis/creation

## Manual installation
### Download the following libraries :
* Pillow
* Numpy
* Tensorflow
* Matplotlib

Or simply run 
```bash
pip install -r requirements.txt 
```

If you run on GPU, do not forget to install cudatoolkit and cudnn.

## The models

* ***GAN*** : this model tries to mimic the input images
* ***Pix2pix*** model : this model translate images (from maps to ground view for instance)
* ***CycleGan*** model : transform the first class into the second (apple to orange for instance)
* ***classifier*** : simply classify the input data into its predictive classe
* ***conditionalGan*** : like GAN, create images, but this time, you can specify the desired class

## Organize your data 

You have to create a specific `folder` containing your data. In this folder, arrange your data in a `train` and `test` directory. 
* For the ***Gan*** and ***Pix2pix*** models, `A_`*.extension is the input while `B_`*.extension is the target. For the basic GAN, only `A_`*.extension are required.
* For the ***CycleGan***, place two different folders (classA and classB for instance) in the train folder.
* For the ***classifier*** and ***conditionalGan***, the images need to contain their classes in their name as such : {classe_name}_*.extension .

## Run the program

Once the `working directory` contains all the data and is well organised, run the following : 
```bash
cd {main_directory}
python main.py {name_workDir} {phase} {name_model}
```
While the name of the model is not required, the others are. The basic model is ***Pix2pix***.

## Your results

The `main.py` file (if in the training phase) will then create saves of your model and outputs of some images during its training. 
The model will be saved in the `__model__` directory and the outputs in the `outputs` directory.

## Use it to predict

You simply have to re-run the same code as above, and specify *test* for the phase. Obviously, the model will be loaded from the directory and use the test directory to create outputs, that will be stored in the `outputs` directory, once again.

### Inspiration 

These models have been creating following these tutorials : 
* [Gan](https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-cifar-10-small-object-photographs-from-scratch/)
* [cycleGan](https://machinelearningmastery.com/cyclegan-tutorial-with-keras/)
* [Pix2pix](https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/)
* [ConditionalGan](https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/)



