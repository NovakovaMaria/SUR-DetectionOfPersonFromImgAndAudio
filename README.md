# SUR project 2023/2024 - Audio and Image Classifier
#### Mária Novákova (xnovak2w), Diana Maxima Držíková (xdrzik01)

The goal of this project is to create classifiers which can detect one specific person in the provided dataset. 
Two models were created - GMM for audio classification and CNN for image classification. 

Predictions for evaluated data: 
- for images = **models/CNN/trained/predictions_img.txt** 
- for audio = **models/GMM/trained/predictions_audio.txt** 

## Audio Classifier - GMM

We created the Gaussian Mixture Model for audio classification. The model contains two Gaussians - one for the target and non-target.
Files are using library **ikrlib.py**, which was modified to match python3.10. 

Three files were created and are handled like this:

### Training GMM

``` python3.10 trainGMM [--cv]```

where:
- cv = Stands for the bool parameter which allows cross-validation to be executed. If not called, the default is *false*.

The models after training are stored in **models/GMM/new** as **gmm_target_model.pkl** and **gmm_nontarget_model.pkl**.

**Put SUR_projekt2023-2024 in root folder**.

### Evaluating GMM

``` python3.10 evaluateGMM [--input filepath] [--output filepath] [--target filepath] [--non_target filepath]```

where:
- input = Input files path that are evaluated. Default *../SUR_projekt2023-2024_eval*.
- output = Output file path where predictions are stored. Default **../models/GMM/new/predictions_audio.txt**.
- target = File with parameters of Gaussian for the target. Default *../models/GMM/trained/gmm_target_model.pkl*.
- non_target = File with parameters of Gaussian for non-target. Default *../models/GMM/trained/gmm_nontarget_model.pkl*.

**Put SUR_projekt2023-2024_eval in root folder**.

### Library for GMM

``` libGMM.py ``` is not supposed to be called on its own. It contains helper functions for training and plotting the results.

## Image Classifier - CNN

We created a Convolutional Neural Network for image classification. 

Three files were created and are handled like this:

### Training CNN

``` python3.10 trainCNN [--threshold float]```

where:
- float = Threshold for hard decision. Default *0.5*.

The models after training are stored in **models/CNN/new** as **model.pth**.

**Put SUR_projekt2023-2024 in root folder**.

### Evaluating CNN

``` python3.10 evaluateCNN [--input filepath] [--output filepath] [--model filepath] [--threshold float]```

where:
- input = Input files path that are evaluated. Default *../SUR_projekt2023-2024_eval*.
- output = Output file path where predictions are stored. Default **../models/CNN/new/predictions_img.txt**.
- model = File with CNN model Default *../models/CNN/new/model.pth*.
- threshold = Threshold for hard decision. Default *0.5*.

**Put SUR_projekt2023-2024_eval in root folder**.



### Library for CNN

``` libCNN.py ``` is not supposed to be called on its own. It contains helper functions for training and plotting the results.
