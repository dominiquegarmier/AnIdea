# AnIdea

## About

This is the git repository for the Maturity Thesis **AnIdea: Approximation der Löung einer Differentialgleichung mit neuronalent Netzwerken** by **Dominique F. Garmier**

originally submited in 2019.

### AnIdea

AnIdea is a neural network based numeric algorithm for solving the initial value problem for non-linear differential equations of first order. As shown in the [paper](./paper/Maturaarbeit_Dominique_Garmier_AnIdea.pdf) the algorithm is able to outperform pre-existing algorithms for specific conditions.

This repository was originally created in 2019. For sake of better readablitiy the repository was restructured in late 2021, without chaning any of the major components.

### Awards

#### Rotary Prize

in 2020 this thesis was awarded the prize for **"best maturity thesis at Kantonsschule Wohlen for the year 2019"**

#### Pro Argovia

in 2020 this thesis was recognized as **"one of the five best maturity theses in the Kanton of Aargau"** (region of Switzerland)

#### Schweizer Jugend Forscht (Swiss Youth Research)

in 2020 this thesis recieved the 2nd highest possible mark **"very good"** in the national contest 

### Use

#### requirements

compatible python version: `3.6`

download repo

```
$ git clone git@github.com:DominiqueGarmier/AnIdea.git
$ cd anidea
```

install the dependencies

```
$ virtualenv .venv p=python3.6
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

#### running the algorithm

switch directory and run python on `run.py`
```
$ cd anidea
$ python run.py *args*
```

#### Running Modes

##### Version
```
$ python run.py
```
prints version and logo

##### Help
```
$ python run.py -h
```
prints help and all possible arguments

##### Others

You can try to run these commands:

```
$ python run.py -t --instancename *some name* --modelname OptimalHyperparameters
```

trains an new instance using the optimal hyperparameters

```
$ python run.py -e --instancename BestResult --plotall
```
creates the most important plots of BestResult (take a look at the config to use the values you want, use h=0.15056 for optimal accuracy) (set 'toPng' to false in config to get direct matplotlib output) (add --plotst to plot e(h) scatter plot, though this will take a while)

```
$ python run.py -l --instancename BestResult
```
continues training BestResults (make sure epochs isnt set to 30,000 in config, else it will end imediatly)

```
$ python run.py -t
```
trains standard instance with random name

```
$ python run.py -t --instancename HelloWorld --modelname HelloWorld
```
trains an example instance

```
$ python scheduler.py
```
(advanced) this will schedule a batch of in this case 5 instances (with different hyperparamters) to be trained and evaluated one after an other. Edit the .py file for more options.

```
$ python tools/FPOs.py
```
this is the scrip used to calculate the FPOs of the NN. Again edit the .py file to see what is going on.
