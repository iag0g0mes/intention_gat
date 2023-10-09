# Interaction-aware Maneuver Intention Prediction for Autonomous Vehicles using Interaction Graphs

Created by Iago Pachêco Gomes at USP - ICMC, University of São Paulo - Institute of Mathematics and Computer Science


**(published in 2023 IEEE Intelligent Vehicles Symposium (IV))**

## Introduction

This repository contains the implementation of the models proposed and evaluated in the article "Interaction-aware Maneuver Intention Prediction for Autonomous Vehicles using Interaction Graphs". In addition, it also provides the lateral and longitudinal maneuver intention labels for the Argoverse v.1 validation dataset.


## Abstract


Intention prediction is an important task for an autonomous vehicle's perception system. It gives the likelihood of a target vehicle performing a maneuver belonging to a finite set of possibilities.  There are many factors that influence the decision-making process of a driver, which should be considered in a prediction framework. In addition, the lack of labeled large-scale dataset with maneuver intention annotation  imposes another challenge to the research field. In this sense, this paper proposes an Interaction-aware Maneuver Intention Prediction framework using interaction graphs to extract complex interaction features from traffic scenes. In addition, we explored a Semi-Supervised approach called Noisy Student to take advance of unlabeled data in the training step. Finally, the results show relevant improvement while using unlabeled data that improved the framework performance.


## System Architecture

![Alt System Architecture](/image/model.png)

This architecture relies on an encoder-decoder setup, where motion, road geometry, and interaction features are extracted from the interaction graph and a High-Definition Map (HD-Map). A Bidirectional-LSTM combines all features, and two decoders with Fully-Connected (FC) layers and Multi-Head Attention Mechanism (MHAM) estimate the lateral and longitudinal maneuver intentions. 

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
##### Extract Features
1) activate the environment and move to the feature folder
   
```shell
conda activate argo
cd features/argoverse
```

2) Modify the file cfg/features.ini according to instructions and the example
3) You have to run this code one time for the validation dataset and another for the training dataset

```python
python argoverse_features.py --cfg cfg/features.ini
```

4) Split the validation dataset into train_with_labels / validation / testing
4.1) Modify the file cfg/split_dataset.ini according to instructions and the example

```python
python split_features.py --cfg cfg/split_dataset.ini
```


### Training and Testing

#### Baseline

#### Noisy Student

#### I-GAT Predictor

## Contact

If you find any bug or issue of the software, please contact 'iagogomes at usp dot br' or 'iago.pg00 at gmail dot com'


