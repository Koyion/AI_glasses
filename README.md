# AI_glasses
Neural Net for glasses\sunglasses\perosn predictions with TensorFlow.

Folder \Glass_predict
model.py - for training and saving the model
continue_train.py - for restore model and continue training
model_functions.py and nn_functions.py - functions of the model
one_image_test.py, folder_image_test.py and hp_open.py - for testing

Folder \image_augument_scroller contains scripts for completing train\test set of images in h5py file.

Total images in train test set used ~ 9500.

Model saved in datasets\BEST-b003la01ep30tra94tea765 is the best for now 29.01.2019.
(L2 regularization beta = 0.003, learning rate = 0.01, num epochs = 30, train accuracy ~ 94%, test accuracy ~ 77%)

5-layers model:
X - input features 256x256x3--> 

--> CONV2D(4x4, 8 filters, stride = 1, padding - same) --> ReLU --> AVGPOOL(4x4, stride = 2, padding - same)  -->

--> CONV2D(4x4, 16 filters, stride = 2, padding - same) --> ReLU --> MAXPOOL(4x4, stride = 4, padding - same) -->
        
--> CONV2D(2x2, 32 filters, stride = 1, padding - same) --> ReLU --> AVGPOOL(2x2, stride = 4, padding - same) -->

--> FLATTEN LAYER -->

--> FULLY CONNECTED LAYER, 160 outputs --> FULLY CONNECTED LAYER, 3 outputs --> softmax
