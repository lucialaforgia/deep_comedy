# AI for generating poetry! :robot: :writing_hand:

<p align="center">
  <img width=500 src="https://github.com/lucialaforgia/deep_comedy/blob/master/images/dante.jpg?raw=true">
</p>


This repository contains some experiments with Recurrent Neural Networks aimed to reproduce the Dante's writing style and generate new text according to it. Our approach starts with some baseline sequence-to-sequence models performing training and generation by characters, by syllables and by words. 
Going deeper and enjoing the project, we have developed a pair of more advanced models, which give us quite impressive results writing text that follows Dante's hendecasyllable, triplets structure and rhyming scheme. More details are explained [here](docs/deep_comedy_documentation.pdf).

**Watch our model writing a new canto one verse after the other!**

<p align="center">
  <img width=700 src="https://github.com/lucialaforgia/deep_comedy/blob/master/images/gen_tonedrev.gif">
</p>


## Getting started

* Clone repository
```
git clone https://github.com/lucialaforgia/deep_comedy.git
```

* Install dependencies (**We recommend Python 3.8 version**)
```
sudo apt install python3.8 python3-setuptools python3-pip python3-venv graphviz
```

* Create a virtual environment and install requirement modules
```
cd deep_comedy
python3 -m venv venv
source venv/bin/activate

pip3 install -r requirements.txt
```


## Running

### Download pretrained models
Our pretrained models can be found on [releases page](https://github.com/lucialaforgia/deep_comedy/releases/tag/pretrained_models). To download all the models you can run the `download_pretrained_models.py` script that will place all the file in to the correct folder.

```
python3 download_pretrained_models.py
```

### Training
All the training scripts are placed in `training_scripts` folder. If you want to re-train a neural network you need to run the corresponding script, listed below:. 
```
training_scripts/train_by_char.py
training_scripts/train_by_syl.py
training_scripts/train_by_word.py
training_scripts/train_by_rev_syl_rhyme.py  
training_scripts/train_by_rev_syl_verse.py
training_scripts/train_by_tonedrev_syl_tone.py
training_scripts/train_by_tonedrev_syl_rhyme.py
training_scripts/train_by_tonedrev_syl_verse.py
```

Our advanced models are composed by more than one neural network, so there is a script to train each neural network. All data will be saved in the `logs` and `models` subfolders in the main model's folder.

For example to train the model `by_char` you need to run:

```
python3 training_scripts/train_by_char.py
```

### Generating
To use the models and generate a new canto you can run one of the following scripts from `generating_scripts` folder, depending on which model you have chosen:
```
generating_scripts/generate_by_char.py
generating_scripts/generate_by_syl.py
generating_scripts/generate_by_word.py
generating_scripts/generate_by_rev_syl.py
generating_scripts/generate_by_tonedrev_syl.py
```
For example to write a new canto with our best model you need to run:
```
python3 generating_scripts/generate_by_tonedrev_syl.py
```

### Evaluation's metrics

We also develop some metrics to evaluate text, which give us the following scores on the first canto of the Divine Comedy, useful to understand the goodness of our generated cantos comparing the values:


| Metric                          | Score          |
| :---                            |      :---:     |
| Number of verses                |       136      |
| Number of strophes              |        46      |
| Number of well formed terzine   |        45      |
| Last single verse               |      True      |
| Average syllables per verse     |  11.07 ± 0.41  |
| Hendecasyllables by tone        |     0.9044     |
| Rhymeness score                 |     0.9710     |


You can quckly calculate all of them running the `evaluate_metrics.py` script, passing as argument a `.txt` file contains one canto.
```
python3 evaluate_metrics.py generated_cantos/generated_canto.txt
```

**Here the metrics computed on a new and machine-written canto:**

| Metric                          | Score          |
| :---                            |      :---:     |
| Number of verses                |       136      |
| Number of strophes              |        46      |
| Number of well formed terzine   |        45      |
| Last single verse               |      True      |
| Average syllables per verse     |  11.07 ± 0.41  |
| Hendecasyllables by tone        |     0.9044     |
| Rhymeness score                 |     0.9710     |


