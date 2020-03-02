# AnIdea (c) 2019 Dominique F. Garmier All Rights Reserved
# Version: pre2.1
# --------------------------------------------------------

import os
import numpy as np
import tensorflow as tf

import src.utils as utils
import src.numerics as numerics
import src.machinelearning as ml
import src.plots as plots


def evaluateInstance(config, instanceName, args):
    ''' evaluates an instance with metric given by args
        
        returns None when done.
        
        will trow errors if instance doesnt exist
        '''

    outputPath = config['output_path']
    
    path = outputPath + '/' + instanceName + '/instance.txt'
    hyperparameters = utils.readModel(path)

    # recovering instance
    instancePath, dataPath, plotsPath, savesPath, previewPath, tmpLogPath, tmpEpochsPath, batchPath = utils.loadInstance(instanceName, outputPath)

    # temporaray files to write to, in case of crash
    tmpLog = open(tmpLogPath, 'a+')

    # define the model
    utils.printToLog("building model...", True, tmpLog)
    lr, t, ic, N_r, dN_r, ddN_r, loss, show_loss, error, optimizer, init, saver, gpu_options = ml.buildModel(hyperparameters, config['omega'])


    # get last checkpoint
    ckpts = os.listdir(savesPath)
    isNew = False
    if len(ckpts) > 1:
        lastCkpt = ckpts[-2]
        ckptPath = savesPath + '/' + lastCkpt.split('.')[0]
    else:
        isNew = True


    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        # init 
        utils.printToLog("started session...", True, tmpLog)

        if isNew:
            sess.run(init)
            utils.printToLog("initialized session", True, tmpLog)
        else:
            saver.restore(sess, ckptPath)
            utils.printToLog("loaded session", True, tmpLog)


        # run different metrics

        if args.plotnn or args.plotall:

            utils.printToLog("calculating nn plot...", True, tmpLog)

            x0 = config['eval_inital_conditions'][0]
            v0 = config['eval_inital_conditions'][1]
            t0 = config['eval_inital_conditions'][2]

            stepsize = config['eval_stepsize']
            stepres = config['eval_stepresolution']
            tRange = config['eval_t_range']

            toDat = config['toDat']
            toPng = config['toPng']

            x_rk4, t_rk4 = numerics.PendulumRK4(x0, v0, t0, 1, tRange, 0.001)

            x_NN = np.array([])
            xdot_NN = np.array([])
            error_NN = np.array([])
            t_NN = np.linspace(t0, t0 + tRange, int(tRange/stepsize)*stepres, endpoint=False)

            for step in range(int(tRange/stepsize)):

                xOffset = x0 - (((x0 + np.pi)%(2*np.pi)) - np.pi)
                x0 = (((x0 + np.pi)%(2*np.pi)) - np.pi)
                
                # sign = np.sign(v0)
                # x0 = (((x0 + np.pi)%(2*np.pi)) - np.pi)*sign
                # v0 = v0*sign

                icBatch = np.reshape([[x0, v0]], (2, 1))
                tBatch = np.reshape(np.linspace(0, stepsize, stepres + 1), (1, -1, 1))

                x_part, x_dot_part, error_part = sess.run([N_r[0,:,0], dN_r[0,:,0], error[0,:,0]], feed_dict={ic:icBatch, t:tBatch})

                x_part = xOffset + x_part
                x_dot_part = x_dot_part

                x0 = x_part[-1]
                v0 = x_dot_part[-1]

                x_NN = np.append(x_NN, x_part[:-1])
                xdot_NN = np.append(xdot_NN, x_dot_part[:-1])
                error_NN = np.append(error_NN, error_part[:-1])

            # plotting
            plots.plotNN(tRange, x_NN, xdot_NN, error_NN, t_NN, x_rk4, t_rk4, plotsPath, dataPath, toDat, toPng)

        if args.plotps or args.plotall:

            utils.printToLog("calculating ps loss plot...", True, tmpLog)

            stepsize = hyperparameters['t_range']
            stepres = config['eval_stepresolution']
            toDat = config['toDat']
            toPng = config['toPng']

            x0_res = 100
            v0_res = 100
            v0_cutoff = 3

            x0_batch = np.linspace(-np.pi, np.pi, x0_res)
            v0_batch = np.linspace(-v0_cutoff, v0_cutoff, v0_res)
            # v0_batch = np.linspace(0, v0_cutoff, v0_res)

            icBatch = np.reshape(np.array([np.broadcast_to(x0_batch, (v0_res, x0_res)), np.transpose(np.broadcast_to(v0_batch, (x0_res, v0_res)))]), (2, x0_res * v0_res))

            _ = np.linspace(0, stepsize, stepres, endpoint=False)
            tBatch = np.reshape(_, (1, stepres, 1))

            error_NN = sess.run(error, feed_dict={t:tBatch, ic:icBatch})
            phasespaceLoss = np.log10(np.reshape(np.mean(np.square(error_NN), axis=1), (v0_res, x0_res)))

            # plotting
            plots.plotPhasespaceLoss(phasespaceLoss, v0_res, x0_res, v0_cutoff, plotsPath, dataPath, toDat, toPng)

        if args.plotls or args.plotall:

            utils.printToLog("calculating loss plot...", True, tmpLog)

            toDat = config['toDat']
            toPng = config['toPng']
            epochs_path = instancePath + '/epochs.anidea'

            file = open(epochs_path, 'r')

            epochs = []
            eval_loss = []
            eval_logloss = []
            timestamps = []

            for line in file:
                vals = line.split(',')

                epochs.append(float(vals[0]))
                eval_loss.append(float(vals[1]))
                eval_logloss.append(float(vals[2]))
                timestamps.append(float(vals[3]))

            file.close()
            epochs = np.array(epochs)
            eval_loss = np.array(eval_loss)
            eval_logloss = np.array(eval_logloss)
            timestamps = np.array(timestamps)

            # plotting
            plots.plotLogLoss(eval_loss, epochs, plotsPath, dataPath, toDat, toPng)

        if args.plotst:

            utils.printToLog("calculating stepsize plot...", True, tmpLog)

            toDat = config['toDat']
            toPng = config['toPng']

            h_min = 0.05
            h_max = 1
            h_steps = 10

            x0_ = config['eval_inital_conditions'][0]
            v0_ = config['eval_inital_conditions'][1]
            t0_ = config['eval_inital_conditions'][2]

            t_range = 5 #config['eval_t_range']
            stepres = 1

            # Baseline with RK4
            #x_rk4, t_rk4 = numerics.PendulumRK4(x0_, v0_, t0_, 1, t_range, 0.001)
            #dx_rk4 = (x_rk4[-1] - x_rk4[-2])/(t_rk4[-1] - t_rk4[-2])
            #rk4_energy = x_rk4[-1] + 0.5*dx_rk4**2 # energy at t_max

            e0 = 1 - np.cos(x0_) + 0.5*v0_**2

            stepsizes = np.exp(np.log(h_min) + ((np.log(h_max) - np.log(h_min)) * np.random.rand(h_steps)))
            errorofh = []

            counter = 0

            utils.printToLog("stepsize loop...", True, tmpLog)
            for stepsize in stepsizes:
                counter += 1

                x0 = x0_
                v0 = v0_

                for i in range(int(t_range/stepsize)):
                    
                    xOffset = x0 - (((x0 + np.pi)%(2*np.pi)) - np.pi)
                    # x0 = (((x0 + np.pi)%(2*np.pi)) - np.pi)
                    # v0 = v0
                    sign = np.sign(v0)
                    x0 = (((x0 + np.pi)%(2*np.pi)) - np.pi)*sign
                    v0 = v0*sign

                    icBatch = np.reshape([[x0, v0]], (2, 1))
                    tBatch = np.reshape(np.linspace(0, stepsize, stepres + 1), (1, -1, 1))
        
                    x_part, x_dot_part, error_part = sess.run([N_r[0,:,0], dN_r[0,:,0], error[0,:,0]], feed_dict={ic:icBatch, t:tBatch})

                    #x0 = xOffset + x_part[-1]
                    #v0 = x_dot_part[-1]

                    x0 = xOffset + x_part[-1]*sign
                    v0 = x_dot_part[-1]*sign

                N_energy = 1 - np.cos(x0) + 0.5*v0**2
                errorofh.append(np.abs(N_energy - e0))

                utils.printToLog("stepsize plot progress: " + str(100*counter / len(stepsizes)) + "%" , True, tmpLog)

            e_plot = np.array(errorofh)
            h_plot = stepsizes

            # plotting
            plots.plotStepsizeScatter(h_plot, e_plot, False, toDat, toPng, plotsPath, dataPath)
            
        # add more metrics

    # write to log files
    utils.printToLog("done!", True, tmpLog)
    utils.fromTmpTo(instancePath)
    return

