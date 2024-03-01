# AnIdea (c) 2019 Dominique F. Garmier All Rights Reserved
# Version: pre2.1
# --------------------------------------------------------

import numpy as np

import src.utils as utils
import src.plots as plots

def visualizeDat(config, dataName, instanceName, args):

    outputPath = config['output_path']
    instancePath, dataPath, plotsPath, savesPath, previewPath, tmpLogPath, tmpEpochsPath, batchPath = utils.loadInstance(instanceName, outputPath)

    datPath = dataPath + '/' + dataName

    if args.plotst:

        toPng = config['toPng']

        lines = open(datPath, 'r').readlines()
        data = []
        for line in lines[1:]:

            vals = list(map(float, line.split()))
            data.append(vals)

        data = np.array(data)

        h_plot = data[:,0]
        e_plot = data[:,1]
        
        # plotting
        plots.plotStepsizeScatter(h_plot, e_plot, False, False, True, plotsPath, dataPath)
