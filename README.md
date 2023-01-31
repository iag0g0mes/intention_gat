# Interaction-aware Maneuver Intention Prediction for Autonomous Vehicles using Interaction Graphs

Created by Iago Pachêco Gomes at USP - ICMC, University of São Paulo - Institute of Mathematics and Computer Science


**(waiting for the result of the submission to 2023 IEEE Intelligent Vehicles Symposium (IV))**

## Introduction

This repository contains the implementation of the models proposed and evaluated in the article "Interaction-aware Maneuver Intention Prediction for Autonomous Vehicles using Interaction Graphs". In addition, it also provides the lateral and longitudinal maneuver intention labels for the Argoverse v.1 validation dataset.


## Abstract


Intention prediction is an important task for an autonomous vehicle's perception system. It gives the likelihood of a target vehicle to perform a maneuver belonging to a finite set of possibilities.  There are many factors that influence the decision-making process of a driver, which should be considered in a prediction framework. In addition, the lack of labeled large-scale dataset with maneuver intention annotation  imposes another challenge to the research field. In this sense, this paper proposes an Interaction-aware Maneuver Intention Prediction framework using interaction graphs to extract complex interaction features from traffic scenes. In addition, we explored a Semi-Supervised approach called Noisy Student to take advance of unlabeled data into the training step. Finally, the results show relevant improvement while using unlabeled data that improved the framework performance.

## License

Apache License 2.0

## Citation
``` 
@article{gomes2023intention,
  title={Interaction-aware Maneuver Intention Prediction for Autonomous Vehicles using Interaction Graphs},
  author={Gomes, Iago Pach{\^e}co and Wolf, Cristiano Premebida, Denis Fernando},
  year={2023}
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


### Training and Testing

#### Baseline

#### Noisy Student

#### I-GAT Predictor

## Contact

If you find any bug or issue of the software, please contact 'iagogomes at usp dot br' or 'iago.pg00 at gmail dot com'


