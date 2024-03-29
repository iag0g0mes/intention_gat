# Interaction-aware Maneuver Intention Prediction for Autonomous Vehicles using Interaction Graphs

Created by Iago Pachêco Gomes at USP - ICMC, University of São Paulo - Institute of Mathematics and Computer Science


**(published in 2023 IEEE Intelligent Vehicles Symposium (IV))**

## Introduction

This repository contains the implementation of the models proposed and evaluated in the article "Interaction-aware Maneuver Intention Prediction for Autonomous Vehicles using Interaction Graphs". In addition, it also provides the lateral and longitudinal maneuver intention labels for the Argoverse v.1 validation dataset.


## Abstract


Intention prediction is an important task for an autonomous vehicle's perception system. It gives the likelihood of a target vehicle performing a maneuver belonging to a finite set of possibilities.  There are many factors that influence the decision-making process of a driver, which should be considered in a prediction framework. In addition, the lack of labeled large-scale dataset with maneuver intention annotation  imposes another challenge to the research field. In this sense, this paper proposes an Interaction-aware Maneuver Intention Prediction framework using interaction graphs to extract complex interaction features from traffic scenes. In addition, we explored a Semi-Supervised approach called Noisy Student to take advance of unlabeled data in the training step. Finally, the results show relevant improvement while using unlabeled data that improved the framework performance.


## System Architecture

![Alt System Architecture](/images/model.png)

This architecture relies on an encoder-decoder setup, where motion, road geometry, and interaction features are extracted from the interaction graph and a High-Definition Map (HD-Map). A Bidirectional-LSTM combines all features, and two decoders with Fully-Connected (FC) layers and Multi-Head Attention Mechanism (MHAM) estimate the lateral and longitudinal maneuver intentions. 

To take advantage of unlabeled data, we employed the Noisy Student approach to estimate pseudo-labels for the training dataset. The results showed that a predictor (with only 2 seconds of obsevation) achieved similar performance than a classifier (which uses 5 seconds of observation - complete scene). 

<p align="center">
<img src="/images/noisy.png" width="700" alt="Noisy Student">
</p>

## License

Apache License 2.0

## Citation
``` 
@inproceedings{gomes2023interaction,
  title={Interaction-aware Maneuver Prediction for Autonomous Vehicles using Interaction Graphs},
  author={Gomes, Iago Pach{\^e}co and Premebida, Cristiano and Wolf, Denis Fernando},
  booktitle={2023 IEEE Intelligent Vehicles Symposium (IV)},
  pages={1--8},
  year={2023},
  organization={IEEE}
}

```

## Usage

### Requirements

- Python 3.8
- scikit-learn 0.23.2 (https://scikit-learn.org/stable/)
- Argoverse API (https://github.com/argoai/argoverse-api)
- TensorFlow 2.8.0

### Features

#### Dataset

1) This project was evaluated in the Argoverse V.1. Motion Forecasting Dataset, available at https://www.argoverse.org/av1.html
2) Follow the instructions in https://github.com/argoverse/argoverse-api to install the argoverse-api
   
##### Extract Features
1) Activate the environment and move to the feature folder
   
```shell
conda activate argo
cd features/argoverse
```

2) Modify the file cfg/features.ini according to instructions and the example
3) You have to run this code one time for the validation dataset and another for the training dataset

```python
python argoverse_features.py --cfg cfg/features.ini
```

4) Modify the file cfg/split_dataset.ini according to instructions and the example
5) Split the validation dataset into train_with_labels / validation / testing

```python
python split_features.py --cfg cfg/split_dataset.ini
```


### Training and Testing

#### Baseline

##### Train

1) Move to the models folder
2) Modify the file cfg/basic_model.ini
   - mode [train] -> for training
4) Run train_basic.py
   
```python
python train_basic.py --cfg cfg/basic_model.ini
```

##### Test
1) Move to the models folder
2) Modify the file cfg/basic_model.ini
   - mode [test]
   - weights_path: folder with the model checkpoints
4) Run train_basic.py

```python
python train_basic.py --cfg cfg/basic_model.ini
```

#### Noisy Student

##### Train
1) Move to the models folder
2) Modify the file cfg/noisy_student.ini
   - Add the checkpoint folder of the teacher model to the attribute  <teacher_dir> in the .ini file  
3) Run train_noisy.py

```python
python train_noisy.py --cfg cfg/noisy_student.ini 
```

4) To train more students, change the [STUDENT-DECODER] and [TEACHER-DECODER] in the .ini file
- [STUDENT-DECODER]: new set of Fully Connected and Dropout layers
- [TEACHER-DECODER]: parameters of the teacher model
- [DIRS] <teacher_dir>: path to the teacher model checkpoint (\<path to the checkpoint>/model)  

##### Test
1) Move to the models folder
2) Modify the file cfg/noisy_student.ini
   - mode [test]
   - weights_path: folder with the model checkpoints
4) Run train_noisy.py

```python
python train_noisy.py --cfg cfg/noisy_student.ini 
```

#### I-GAT Predictor

##### Train
1) Move to the models folder
2) Modify the file cfg/intention.ini   
4) Run train_intention.py

```python
python train_intention.py --cfg cfg/intention.ini 
```
   
##### Test

1) Move to the models folder
2) Modify the file cfg/intention.ini
   - mode [test]
   - weights_path: folder with the model checkpoints
4) Run train_intention.py

```python
python train_intention.py --cfg cfg/intention.ini 
```

## Contact

If you find any bug or issue of the software, please contact 'iagogomes at usp dot br' or 'iago.pg00 at gmail dot com'


