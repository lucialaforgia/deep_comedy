<p align="center">
  <img width=500 src="https://github.com/lucialaforgia/deep_comedy/blob/master/images/dante.jpg?raw=true">
</p>

This repository contains some experiments with Recurrent Neural Networks aimed to reproduce the Dante's writing style and generate new text according to it. Our approach starts with some baselines sequence-to-sequence models performing training and generation by characters, by syllables and by words. 
Going deeper and enjoing the project, we have developed a pair of more advanced models, which give us quite impressive results writing text that follows Dante's hendecasyllable, triplets structure and rhyming scheme. 

Look an example!

<p align="center">
  <img width=800 src="https://github.com/lucialaforgia/deep_comedy/blob/master/images/gen_tonedrev.gif">
</p>


## Getting started

* Clone repository
```
git clone https://github.com/lucialaforgia/deep_comedy.git
```

* Install dependencies (**We recommend Python 3.8 version**)
```
sudo apt install python3.8 python3-setuptools python3-pip python3-venv
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
Our pretrained models can be found on [releases page](https://github.com/lucialaforgia/deep_comedy/releases/tag/pretrained_models). To download all the models you can run the `download_pretrained_models.py` and they will be placed to the correct folder.

```
python3 download_pretrained_models.py
```

### Training
All the training scripts are placed in `training_scripts` folder. If you want to re-train a neural network you need to run the corresponding script, listed below:. 
```
train_by_char.py
train_by_syl.py
train_by_word.py
train_by_rev_syl_rhyme.py  
train_by_rev_syl_verse.py
train_by_tonedrev_syl_tone.py
train_by_tonedrev_syl_rhyme.py
train_by_tonedrev_syl_verse.py
```

Our advanced models are composed by more than one neural network, so there is a script to train each neural network. All data will be saved in the `logs` and `models` subfolders in the main model's folder.

For example to train the model `by_char` you need to run:

```
python3 training_scripts/train_by_char.py
```

### Generation
To use the models and generate a new canto you can run one of the following scripts from `generating_scripts` folder, depending on which model you have chosen:
```
generating_scripts/generate_by_char.py
generating_scripts/generate_by_syl.py
generating_scripts/generate_by_word.py
generating_scripts/generate_by_rev_syl.py
generating_scripts/generate_by_tonedrev_syl.py
```
For example to write a nre canto with our best generation model you need to run:
```
python3 generating_scripts/generate_by_tonedrev_syl.py
```
