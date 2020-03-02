# AnIdea (c) 2019 Dominique F. Garmier All Rights Reserved
# Version: pre2.1
# --------------------------------------------------------


# imports
import os
import sys
import ast

import datetime as dt
import numpy as np

def readConfig(doWipe):
    '''returns config as dictionary'''

    default_model = [
    [1, 'input layer'], # input shape
    [10, 'p2'],
    [10, [2, 2, 2]],
    [10, [2, 2, 2]],
    [10, [2, 2, 2]],
    [10, [2, 2, 2]],
    [10, 'p2'],
    [1, 'output layer']
    ]

    config_path = "./config.txt"
    default_config = {'output_path':'./out',
                      'models_path':'./models',
                      'do_preview':True,
                      'epochs':30000,
                       'backup_frequency': 10000,
                       'omega': 1,
                       'default_hyperparams':{'model_architecture': default_model,
                                                'loss_function': 'linloss',
                                                't_batchsize': 100,
                                                't_range':0.5,
                                                'bc_batchsize': 100,
                                                'bc_variance': 2,
                                                'learningrate': 0.0001,
                                                'batch_grouping': 10},
                       'eval_inital_conditions':[2,0,0],
                       'eval_stepsize':0.15,
                       'eval_stepresolution':10,
                       'eval_t_range':20,
                       'toDat':True,
                       'toPng':True
                       }

    # write config
    if not os.path.isfile(config_path) or doWipe:

        cfg = open(config_path, 'w+')
        cfg.write(str(default_config))
        cfg.close()
        return default_config

    else:

        cfg = open(config_path, 'r')
        config = ast.literal_eval(cfg.read())
        cfg.close()
        return config


def makeNewInstance(output_path, hyperparams, name):

    if name:
        instance_path = output_path + '/' + name

    else:
        instance_code = dt.datetime.now().strftime("%Y%m%d%H%M%S")
        instance_path = output_path + '/' + instance_code

    # attempt to create instance folder
    if not os.path.isdir(instance_path):
        os.mkdir(instance_path)
        #aprint('Successfully created new instance folder at: ' + instance_path)

    else:
        
        raise FileExistsError("Instance " + name + " already exists")
 
    dataPath = instance_path + '/data'
    plotsPath = instance_path + '/plots'
    savesPath = instance_path + '/saves'
    previewPath = instance_path + '/preview'

    tmplog_path = instance_path + '/tmplog.anidea'
    log_path = instance_path + '/log.txt'
    tmpepochs_path = instance_path + '/tmpepochs.anidea'
    epochs_path = instance_path + '/epochs.anidea'
    instancetxt_path = instance_path + '/instance.txt'
    batch_path = instance_path + '/batch.anidea'

    # make folders
    os.mkdir(dataPath)
    os.mkdir(plotsPath)
    os.mkdir(savesPath)
    os.mkdir(previewPath)

    # make txt files
    _ = open(tmplog_path, 'w+')
    _.close()
    _ = open(log_path, 'w+')
    _.close()
    _ = open(tmpepochs_path, 'w+')
    _.close()
    _ = open(epochs_path, 'w+')
    _.close()
    _ = open(batch_path, 'w+')
    _.close()
    _ = open(instancetxt_path, 'w+')
    _.write(str(hyperparams))
    _.close()

    return instance_path, dataPath, plotsPath, savesPath, previewPath, tmplog_path, tmpepochs_path, batch_path


def loadInstance(name, output_path):

    instance_path = output_path + '/' + name

    if not os.path.isdir(instance_path):
        raise FileNotFoundError("instance " + name + " does not exist!")

    dataPath = instance_path + '/data'
    plotsPath = instance_path + '/plots'
    savesPath = instance_path + '/saves'
    previewPath = instance_path + '/preview'

    tmplog_path = instance_path + '/tmplog.anidea'
    tmpepochs_path = instance_path + '/tmpepochs.anidea'
    batch_path = instance_path + '/batch.anidea'

    return instance_path, dataPath, plotsPath, savesPath, previewPath, tmplog_path, tmpepochs_path, batch_path


def printEpoch(loss, epoch, tmpepochs):
    timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")

    s_ = str(epoch) + ", " + str(loss) + ", " + str(np.log(loss)) + ", " + timestamp + "\n"
    tmpepochs.write(s_)

 
def printToLog(string, toConsole, tmplog):

    timestamp = str(dt.datetime.now())
    s_ = "AnIdea: " + timestamp + " : "  + string
    if toConsole:
        print(s_)
    tmplog.write(s_ + "\n")


def fromTmpTo(instance_path):

    # log
    tmplog = open(instance_path + '/tmplog.anidea', 'r')
    log = open(instance_path + '/log.txt', 'a+')

    # append tmp to
    log.write(tmplog.read())
    log.close()
    tmplog.close()
    
    # erase tmp file
    open(instance_path + '/tmplog.anidea', 'w').close()

    # epochs
    tmpepochs = open(instance_path + '/tmpepochs.anidea', 'r')
    epochs = open(instance_path + '/epochs.anidea', 'a+')

    # append tmp to
    epochs.write(tmpepochs.read())
    epochs.close()
    tmpepochs.close()
    
    # erase tmp file
    open(instance_path + '/tmpepochs.anidea', 'w').close()

def readModel(path):
    
    if not os.path.isfile(path):
        raise FileNotFoundError(str(path) + ' does not exist')

    temp = open(path, 'r').read()

    model_architecture = ast.literal_eval(temp)
    return model_architecture


def saveBatch(batch_path, t_batch, bc_batch):
    s = str([t_batch.tolist(), bc_batch.tolist()])
    open(batch_path, 'w').close()
    _ = open(batch_path, 'w+')
    _.write(s)
    _.close()
    return


def readBatch(batch_path):
    _ = open(batch_path, 'r')
    s = _.read()
    arr = ast.literal_eval(s)

    t_batch = np.array(arr[0])
    bc_batch = np.array(arr[1])

    return t_batch, bc_batch