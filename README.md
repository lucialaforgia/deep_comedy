<p align="center">
  <img width=512px src="https://github.com/lucialaforgia/deep_comedy/blob/master/images/dante.jpg?raw=true">
</p>


<p align="center">
  <img width=812px src="https://github.com/lucialaforgia/deep_comedy/blob/master/images/gen_tonedrev.gif">
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

* **Download pretrained models**
Our pretrained models can be found on [releases page](https://github.com/lucialaforgia/deep_comedy/releases/tag/pretrained_models). To download all the models you can run the `download_pretrained_models.py` and they will be placed to the correct folder.

```
python3 download_pretrained_models.py
```

* **Training:** All the training scripts are placed in `training_scripts` folder. If you want to re-train a neural network you need to run the corresponding script and all data will be saved in the `logs` and `models` subfolders in the main project's folder.

```
python3 training_scripts/train_by_char.py
```

* **Generation:** To resume last saved training and continue it.

```
python3 generating_scripts/generate_by_tonedrev_syl.py
```
