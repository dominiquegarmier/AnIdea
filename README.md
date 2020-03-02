AnIdea (c) 2019 Dominique F. Garmier All Rights Reserved
Version: pre2.1
--------------------------------------------------------
REQUIREMENTS:

numpy==1.16.3
matplotlib==3.0.3
tensorflow==1.13.1

FOLDER STRUCTURE:

please make sure these files and folders are in their correct location relative to ROOT

ROOT---models
	 |-out
	 |
	 |-src--machinelearning.py
	 |	  |-numerics.py
	 |	  |-plots.py
	 |	  |-tfopsbatchjac.py
	 |	  |-utils.py
	 |
	 |-tools-FPOs.py
	 |
	 |-run.py
	 |-evaluate.py
	 |-training.py
	 |-vis.py
	 |
	 |-(scheduler.py)



TO RUN:

run python in ROOT-folder (same as run.py) and call run.py


try these: (to stop training: either wait for the 30,000 Epochs to be over or press ctrl+c, this shouldn't break anything)

- run.py -h 																: for help on the possible args

- run.py 																	: prints the logo and version

- run.py -t --instancename *some name* --modelname OptimalHyperparameters	: trains an new instance using the optimal hyperparameters

- run.py -e --instancename BestResult --plotall								: creates the most important plots of BestResult (take a look at the config to use the values you want, use h=0.15056 for optimal accuracy) (set 'toPng' to false in config to get direct matplotlib output) (add --plotst to plot e(h) scatter plot, though this will take a while)

- run.py -l --instancename BestResult										: continues training BestResults (make sure epochs isnt set to 30,000 in config, else it will end imediatly)

- run.py -t 																: trains standard instance with random name

- run.py -t --instancename HelloWorld --modelname HelloWorld 				: trains an example instance

- scheduler.py																: (advanced) this will schedule a batch of in this case 5 instances (with different hyperparamters) to be trained and evaluated one after an other. Edit the .py file for more options.

- tools/FPOs.py																: this is the scrip used to calculate the FPOs of the NN. Again edit the .py file to see what is going on.
