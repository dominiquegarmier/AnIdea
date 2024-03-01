# AnIdea (c) 2019 Dominique F. Garmier All Rights Reserved
# Version: pre2.1
# --------------------------------------------------------

import os
import sys
import ast
import argparse


# AnIdea Imports
#
import training
import evaluate
import vis
import src.utils as utils
#
#---------------


if __name__ == '__main__':

    # Arguments Parser

    parser = argparse.ArgumentParser()

    # Run Modes
    modes = parser.add_mutually_exclusive_group()

    modes.add_argument('-t', '--trainnew', action='store_true', help='create and start training new instance')
    modes.add_argument('-l', '--loadold', action='store_true', help='load previous instance and contiune training')
    modes.add_argument('-e', '--evaluate', action='store_true', help='load and evaluate previous instance')
    modes.add_argument('-p', '--plot', action='store_true', help='*work in progress* load .dat file and plot it')

    # global args
    parser.add_argument('-s', '--silent', action='store_true', help='run AnIdea in silent mode without any text output')
    parser.add_argument('-d', '--debug', action='store_true', help='run AnIdea in debug mode for testing')
    parser.add_argument('--wipeconfig', action='store_true', help='wipe the AnIdea conifg and restore default')

    # mode related args
    parser.add_argument('--modelname', type=str, help='name of the model')
    parser.add_argument('--instancename', type=str, help='name of the instance')
    parser.add_argument('--dataname', type=str, help='path of .dat file to plot')
    parser.add_argument('--plotnn', action='store_true', help='in combination with -evaluate, to plot the nn')
    parser.add_argument('--plotps', action='store_true', help='in combination with -evaluate, to plot the phasespacesloss')
    parser.add_argument('--plotls', action='store_true', help='in combination with -evaluate, to plot the loss over time')
    parser.add_argument('--plotst', action='store_true', help='in combination with -evaluate, to plot the stepsize plot')
    parser.add_argument('--plotall', action='store_true', help='in combination with -evaluate, to plot all the plots but the stepsize')

    # gather args
    args = parser.parse_args()

    # print logo
    print("  █████╗ ███╗   ██╗██╗██████╗ ███████╗ █████╗ ")
    print(" ██╔══██╗████╗  ██║██║██╔══██╗██╔════╝██╔══██╗")
    print(" ███████║██╔██╗ ██║██║██║  ██║█████╗  ███████║")
    print(" ██╔══██║██║╚██╗██║██║██║  ██║██╔══╝  ██╔══██║")
    print(" ██║  ██║██║ ╚████║██║██████╔╝███████╗██║  ██║")
    print(" ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝")
    print("----------------------------------------------")
    print("AnIdea (c) 2019 Dominique F. Garmier All Rights Reserved")
    print("running version: 2.1")
    print("----------------------------------------------")

    # check for correct args
    if args.loadold:
        if not args.instancename:
            exit('Argument Error: Arg --instancename missing')

    if args.evaluate:
        if not args.instancename:
            exit('Argument Error: Arg --instancename missing')

    if args.plot:
        if not args.dataname:
            exit('Argument Error: Arg --path missing')

    # read config
    config = utils.readConfig(doWipe=args.wipeconfig)
    outputPath = config['output_path']
    if not os.path.exists(outputPath):
    	os.makedirs(outputPath)


    if args.trainnew:
        # create new instance and train it
        training.trainNewInstance(config, modelName = args.modelname, instanceName = args.instancename)

    if args.loadold:
        # load existing instance and train it
        training.trainOldInstance(config, instanceName = args.instancename)

    if args.evaluate:
        #run eval
        evaluate.evaluateInstance(config, instanceName = args.instancename, args = args)

    if args.plot:
        #run eval
        vis.visualizeDat(config, dataName = args.dataname, instanceName = args.instancename, args = args)




