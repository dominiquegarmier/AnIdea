# AnIdea (c) 2019 Dominique F. Garmier All Rights Reserved
# Version: pre2.1
# --------------------------------------------------------

import os
import datetime as dt

cfgPath = './config.txt'
runPath = './run.py'
modelsPath = './models'
pyPath = 'py'

tmpModelName = 'tmp' + dt.datetime.now().strftime("%Y%m%d%H%M%S")
tmpModelPath = modelsPath + '/' + tmpModelName  + '.anidea'

def writeModelFile(default_model, modelPath):
    _ = open(modelPath, 'w+')
    _.write(str(default_model))
    _.close()
    return

opts = {'learningrate':[0.1, 0.01, 0.001, 0.0001, 0.00001]}
names =['test1','test2','test3','test4','test5']

start_model = {'model_architecture': [[1, 'input layer'],
    [20, [2,4,4,2]],
    [20, [2,4,4,2]],
    [20, [2,4,4,2]],
    [20, [2,4,4,2]],
    [20, [2,4,4,2]],
    [20, [2,4,4,2]],
    [20, [2,4,4,2]],
    [20, [2,4,4,2]],
    [20, [2,4,4,2]],
    [20, [2,4,4,2]],
    [1, 'output layer']],
    'loss_function': 'pidloss',
    't_batchsize': 40,
    't_range': 0.2,
    'bc_batchsize': 200,
    'bc_variance': 2,
    'learningrate': 0.0001,
    'batch_grouping': 100}
                       

# spell checking
_ = None
for feature in opts:
    if _ and not _ == len(opts[featrue]):
        raise AttributeError('list of states must have same length for all features')
    _ = len(opts[feature])

for i in range(_):

    for feature in opts:

        start_model[feature] = opts[feature][i]

    writeModelFile(start_model, tmpModelPath)
    command = pyPath + ' ' + runPath + ' -t --instancename ' + names[i] + ' --modelname ' + tmpModelName
    os.system(command)
    command = pyPath + ' ' + runPath + ' -e --instancename ' + names[i] + ' --plotall'
    




