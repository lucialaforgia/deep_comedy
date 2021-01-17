# AI for generating poetry! :robot: :writing_hand:

<p align="center">
  <img width=500 src="https://github.com/lucialaforgia/deep_comedy/blob/master/images/dante.png?raw=true">
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


You can quickly calculate all of them running the `evaluate_metrics.py` script, passing as argument a `.txt` file contains one canto.
```
python3 evaluate_metrics.py generated_cantos/generated_canto.txt
```



#### Example of a new and machine-written canto
```
CANTO
fuor de la salatona grave spento
una fuggendo e che tutto mi si spense
come la prima e materia contento

qual è colui che di fuoro e sense
se ch'avrai da me quel che fu assunta
o frate omai convien che tu pense

sì per lo suo parlare a me per punta
così sen giva e non credo che sia
che con le piante potenza defunta

sanza mio cesare a tal melodia
che l'angel poscia dritto e da sguardo
forse qual è legato e poi venia

sì che l'accende domandar più tardo
sì che per digiunar quando fuor porte
nave sanza rispuosi cento sempr' ardo

che mi pareva prima quel corte
oh diss' io e 'l suon de la ventura
che legge amore o lievemente o forte

e viva per nascendo ancor paura
rimossi quando besi tanto baglio
pensa lettor si con l'occhio cura

ingiusto mar di questo angusto vaglio
però tu mai non vo' che tu convienti
nel qual sarà in pergamo tal berzaglio

e strinse mai contrate non altrimenti
che son nel mezzo del troppo si discende
e da ciascun la croce un lume si 'mprenti

che l'animo mio cantar pretende
sì ch'io approvo ciò che più fero e maggio
ma poi che tegna forte a sé comprende

o rimira lor presa e lor vantaggio
di qua dal dolce stil novo ch'i' trova
li occhi rivolsi al suon di questo raggio

quali colombe surga o che si mova
scintillando v'era più la lucerne
l'una parte in te con mirabil prova

a le membra non fa che fa basterne
fecer che dio ma poi che molti amore
o e già le sue bellezze etterne

sempre esser buono operaro a suo fattore
ma sol che 'l teso vi per orlando
da mostrar lo 'nferno a lui colore'

la cui sovresso il nido cominciando
e io mia natura e questo lucano
quali i beati al novissimo bando

perché infino a qui per l'alto mano
quel de la mente in voi che ti mundo corde
fu nobil sia note ma sovrano

parlando fosse stata l'altre corde
quando sarai di là da me non suone
pensava così 'questi per voi non morde

girando e mormorando l'affezione
la bestia ad ogne passo ciò s'accorsi
caccerà l'altra con molta offensione

poscia che pur io di corpo morsi
che posson far lo cor volgere a dio
perch' io parti' così com' io m'accorsi

per ch'ella che vedëa il tacer mio
con tant' ordine rende no ed viva
ed elli a me l'amor com' io

quella milizia che è quel che viva
ch'io là dove più tardi in cielo è torto
noi ricidemmo il cerchio a l'altra riva

se non molto lontan da tra l'orto
se la prima era scempio e ora è tanto
che se m'avresti per venire a porto

badendo sé ne l'altissimo canto
fra me 'dille' dicea 'a la mia donna
e quindi qua tutti a lume santo

ma per avero cinta de la sonna
furon quel che tu abbi però ricorre
che le gamtendi fuor di gonna in gonna

dinanzi a me con imaginare abborre
che sì mi fece picciola vigilia
e la vostra ten va più non soccorre

e poi che de la vendetta la rosquilia
però in fama ne' termini suoi
in una donna di mille milia

quivi s'inganna e dietro ad essa poi
nel viso e 'l dimandar per terminarmi
e lo omai diss' io ciò che vi noi

sorpresi la onona e risparmi
seguitar lo quale io per smeraldi
ciascuna parte fuor de le sue armi

più trista forma che son sì ch'io caldi
e 'l sol san serotini e lucenti
nel vero suo tutto povan saldi

or con altri or con altri reggimenti
d'aguglion fan di cain favoleggiava
fu a quel canto con altri reggimenti

da polimenza di travigliava
pasciute di qua con lascia muover queta
fortunato insieme con l'antica tava

da voi per tepidezza in ben far lieta
e tanto quanto mi lasciò qui cibo
io era novirata in sé asseta

ch'i' fe' di me s'io dissi beatribo
ringrazia il sol de li occhinati e vanti
così di loro angelico caribo

mi fa nascere i fiori e ' frutti santi
par de l'omor che mal fiera crudele
increata voce poi disceser tanti

da la mia state in corpo si disvele
l'impeto suo principio discerna
quell' anima vidi che per noi cele

lo spirito de la fossa sempiterna
non quelli a cui fu rotto il petto e l'ombra
sì di parnaso o bevve in sua cisterna

che non paresse aver la mente ingombra
di questa bestia per la qual tu paresti
ch'è contraposto a quel che 'l ciel t'adombra

de sanza cui suo cristo esser non vesti

```


**Here the metrics computed on the generated canto:**

| Metric                          | Score          |
| :---                            |      :---:     |
| Number of verses                |       112      |
| Number of strophes              |        38      |
| Number of well formed terzine   |        37      |
| Last single verse               |      True      |
| Average syllables per verse     |  11.08 ± 0.55  |
| Hendecasyllables by tone        |     0.7321     |
| Rhymeness score                 |     0.9474     |


