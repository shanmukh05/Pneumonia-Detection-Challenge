# Pneumonia-Detection-Challenge - DPhi
[Check-out the competition here](https://dphi.tech/challenges/pneumonia-classification-challenge-by-segmind/76/overview/about)

[Final Leaderboard - 5th position in Private LB and 7th in Public LB](https://dphi.tech/challenges/pneumonia-classification-challenge-by-segmind/76/leaderboard/private/)

# Problem Statement

Classify chest X-rays(CXRs) with pneumonia from their normal CXR counterparts, using machine learning and computer vision techniques.

# Dataset 

[Dataset and pretrained models are uploaded in Kaggle](https://www.kaggle.com/shanmukh05/pneumonia-classification-challenge)

## Classes
- `Normal`
- `Pneumonia`

## Training Data
- Normal : 1280
- Pneumonia : 1145
- Distribution : 

![image](https://user-images.githubusercontent.com/65073329/123539921-cb6e5980-d759-11eb-963c-db3401c9b21b.png)

## Test Data
- 606 images

# Training

- As the given dataset contains very less instances, training from scratch will not given best results. So I used following pretrained models
   - [ChexNet - keras Implementation](https://github.com/brucechou1983/CheXNet-Keras)
   - [ChexNet - VGG](https://github.com/shanmukh05/Pneumonia-Detection-Challenge/blob/main/papers/CheXNet_VGG.pdf) 

- [TPU as Accelerator](https://github.com/shanmukh05/Pneumonia-Detection-Challenge/blob/main/notebooks/pneumonia-classification-challenge-tpu-training.ipynb)
- [GPU as Accelerator](https://github.com/shanmukh05/Pneumonia-Detection-Challenge/blob/main/notebooks/pneumonia-classification-challenge-gpu-training.ipynb)
- [Best Result](https://github.com/shanmukh05/Pneumonia-Detection-Challenge/blob/main/notebooks/80_61.ipynb)
- [Submission Notebook](https://github.com/shanmukh05/Pneumonia-Detection-Challenge/blob/main/notebooks/pneumonia-classification-challenge-submission.ipynb)

## Best Results
- Metric : `Accuracy`
- Best Results are stored in [experiments.xlsx](https://github.com/shanmukh05/Pneumonia-Detection-Challenge/blob/main/experiments.xlsx). (Versions denotes the version in Kaggle notebook.)
- ChexNet Implementation with freezing bottom layers gave best results (DenseNet121 Implementation)
- Following is the model with highest score (**Private LB : 83.16, Public LB : 80.61**).
``` 
    model = tf.keras.applications.DenseNet121(weights= "imagenet",
                                    include_top=False,
                                    input_shape=(HEIGHT,WIDTH,CHANNELS), pooling="avg")
    predictions = tf.keras.layers.Dense(14, activation='sigmoid', name='predictions')(model.output)
    model = tf.keras.Model(inputs=model.input, outputs=predictions)
    
    model.load_weights("../input/pneumonia-classification-challenge/pretrained.h5")
    model = tf.keras.Model(model.input, model.layers[-2].output)
    
    x = tf.keras.layers.Dense(512, activation = "relu")(model.output)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation = "relu")(x)
    x = tf.keras.layers.Dense(64, activation = "relu")(x)
    outputs = tf.keras.layers.Dense(2, activation = "softmax", dtype = tf.float32)(x)
    model = tf.keras.Model(model.input,outputs)
    
    for layer in model.layers[:-14]:
        layer.trainble = False 
```
