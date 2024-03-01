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

def trainNewInstance(config, modelName, instanceName):
    ''' creates and trains new instance with specified model,
        returns nothing,
        only ends if config.epochs is set to a numerical value
        
        will trow errors if files dont exist as specified'''


    outputPath = config['output_path']
    modelsPath = config['models_path']

    if modelName:
        path = modelsPath + '/' + modelName + ".anidea"
        hyperparameters = utils.readModel(path)
    else:
        hyperparameters = config['default_hyperparams']


    # instance
    instancePath, dataPath, plotsPath, savesPath, previewPath, tmpLogPath, tmpEpochsPath, batchPath = utils.makeNewInstance(outputPath, hyperparameters, instanceName)


    # temporaray files to write to, in case of crash
    tmpLog = open(tmpLogPath, 'a+')
    tmpEpochs = open(tmpEpochsPath, 'w')

    
    # preview rk4
    if config['do_preview']:
        utils.printToLog("calculating rk4...", True, tmpLog)
        prv_t_nn = np.reshape(np.linspace(0, 10, 1000, endpoint=False), (1, 1000, 1))
        prv_ic_nn = np.reshape([1, 1], (2, 1))

        prv_x_rk, prv_t_rk = numerics.PendulumRK4(x0 = 1, v0 = 1, t0 = 0, omega = 1, t_range=10, stepsize = 0.001)
        utils.printToLog("done calculation rk4", True, tmpLog)


    # define the model
    utils.printToLog("building model...", True, tmpLog)
    lr, t, ic, N_r, dN_r, ddN_r, loss, show_loss, error, optimizer, init, saver, gpu_options = ml.buildModel(hyperparameters, config['omega'])


    # learning rate at epoch 0
    lr0 = hyperparameters['learningrate']

    # Session
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        utils.printToLog("started session...", True, tmpLog)
        # init 
        sess.run(init)
        utils.printToLog("initialized session", True, tmpLog)


        # make first batch
        tBatch, icBatch = ml.makeBatch(hyperparameters['t_batchsize'], hyperparameters['t_range'], hyperparameters['bc_batchsize'], hyperparameters['bc_variance'])
        feed_dict_batch = {t:tBatch, ic:icBatch, lr:lr0}


        # training loop vars
        currEpoch = 0
        currLoss = 0

        utils.printToLog("starting training loop...", True, tmpLog)
        while not config['epochs'] or currEpoch < config['epochs']: # if config.epochs = False then this loop will be indefinite

            currEpoch += 1
            lr_e = lr0*(np.e**(-currEpoch/50000)) # lr decay 1/100 at 100k ish
            feed_dict_batch = {t:tBatch, ic:icBatch, lr:lr_e} # change lr_e to lr0 to disable lr decay

            currLoss, _ = sess.run([show_loss, optimizer], feed_dict = feed_dict_batch)

            utils.printToLog("epoch: " + str(currEpoch) + " loss: " + str(currLoss), True, tmpLog)
            utils.printEpoch(currLoss, currEpoch, tmpEpochs)

            if currEpoch % config['backup_frequency'] == 0:

                tmpLog.close()
                tmpEpochs.close()

                utils.fromTmpTo(instancePath)

                tmpLog = open(tmpLogPath, 'a+')
                tmpEpochs = open(tmpEpochsPath, 'a+')


                utils.printToLog("saving model...", True, tmpLog)
                path = savesPath + '/' + str(currEpoch).zfill(12)
                _ = saver.save(sess, path)
                utils.printToLog("saved model to: " + _, True, tmpLog)

                if config['do_preview']:

                    utils.printToLog("making preview plot...", True, tmpLog)
                    # calculating preview nn plot
                    prv_x_nn, prv_error_nn = sess.run([N_r[0,:,0], error[0,:,0]], feed_dict={ic:prv_ic_nn, t:prv_t_nn})
                    

                    # plot preview plot
                    path = previewPath + '/' + str(currEpoch)
                    plots.plotPreview(prv_x_nn, prv_error_nn, prv_t_nn[0,:,0], prv_x_rk, prv_t_rk, path)
                    utils.saveBatch(batchPath, tBatch, icBatch)

            if hyperparameters['batch_grouping'] and currEpoch % hyperparameters['batch_grouping'] == 0:
                tBatch, icBatch = ml.makeBatch(hyperparameters['t_batchsize'], hyperparameters['t_range'], hyperparameters['bc_batchsize'], hyperparameters['bc_variance'])

    return


def trainOldInstance(config, instanceName):
    ''' loads and trains existing instance,
        returns nothing,
        only ends if config.epochs is set to a numerical value
        
        will trow errors if files dont exist as specified'''


    outputPath = config['output_path']
    
    path = outputPath + '/' + instanceName + '/instance.txt'
    hyperparameters = utils.readModel(path)

    # recovering instance
    instancePath, dataPath, plotsPath, savesPath, previewPath, tmpLogPath, tmpEpochsPath, batchPath = utils.loadInstance(instanceName, outputPath)

    # temporaray files to write to, in case of crash
    tmpLog = open(tmpLogPath, 'a+')
    tmpEpochs = open(tmpEpochsPath, 'w')

    # preview rk4
    if config['do_preview']:
        utils.printToLog("calculating rk4...", True, tmpLog)
        prv_t_nn = np.reshape(np.linspace(0, 10, 1000, endpoint=False), (1, 1000, 1))
        prv_ic_nn = np.reshape([1, 1], (2, 1))

        prv_x_rk, prv_t_rk = numerics.PendulumRK4(x0 = 1, v0 = 1, t0 = 0, omega = 1, t_range=10, stepsize = 0.001)
        utils.printToLog("done calculation rk4", True, tmpLog)


    # define the model
    utils.printToLog("building model...", True, tmpLog)
    lr, t, ic, N_r, dN_r, ddN_r, loss, show_loss, error, optimizer, init, saver, gpu_options = ml.buildModel(hyperparameters, config['omega'])


    # learning rate at epoch 0
    lr0 = hyperparameters['learningrate']

    # get last checkpoint
    ckpts = os.listdir(savesPath)
    isNew = False
    if len(ckpts) > 1:
        lastCkpt = ckpts[-2]
        ckptPath = savesPath + '/' + lastCkpt.split('.')[0]
    else:
        isNew = True

    
    # Session
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        # init 
        utils.printToLog("started session...", True, tmpLog)

        if isNew:
            sess.run(init)
            utils.printToLog("initialized session", True, tmpLog)
        else:
            saver.restore(sess, ckptPath)
            utils.printToLog("loaded session", True, tmpLog)

        # batch
        if isNew:
            tBatch, icBatch = ml.makeBatch(hyperparameters['t_batchsize'], hyperparameters['t_range'], hyperparameters['bc_batchsize'], hyperparameters['bc_variance'])
            feed_dict_batch = {t:tBatch, ic:icBatch, lr:lr0}
        else:
            tBatch, icBatch = utils.readBatch(batchPath)
            feed_dict_batch = {t:tBatch, ic:icBatch, lr:lr0}


        # training loop vars

        if isNew:
            currEpoch = 0
        else:
            currEpoch = int(lastCkpt.split('.')[0][-12:])

        currLoss = 0

        utils.printToLog("starting training loop...", True, tmpLog)
        while not config['epochs'] or currEpoch < config['epochs']: # if config.epochs = False then this loop will be indefinite

            currEpoch += 1
            lr_e = lr0*(np.e**(-currEpoch/50000)) # lr decay 1/100 at 100k ish
            feed_dict_batch = {t:tBatch, ic:icBatch, lr:lr_e} # change lr_e to lr0 to disable lr decay

            currLoss, _ = sess.run([show_loss, optimizer], feed_dict = feed_dict_batch)

            utils.printToLog("epoch: " + str(currEpoch) + " loss: " + str(currLoss), True, tmpLog)
            utils.printEpoch(currLoss, currEpoch, tmpEpochs)

            if currEpoch % config['backup_frequency'] == 0:

                tmpLog.close()
                tmpEpochs.close()

                utils.fromTmpTo(instancePath)

                tmpLog = open(tmpLogPath, 'a+')
                tmpEpochs = open(tmpEpochsPath, 'a+')


                utils.printToLog("saving model...", True, tmpLog)
                path = savesPath + '/' + str(currEpoch).zfill(12)
                _ = saver.save(sess, path)
                utils.printToLog("saved model to: " + _, True, tmpLog)

                if config['do_preview']:

                    utils.printToLog("making preview plot...", True, tmpLog)
                    # calculating preview nn plot
                    prv_x_nn, prv_error_nn = sess.run([N_r[0,:,0], error[0,:,0]], feed_dict={ic:prv_ic_nn, t:prv_t_nn})
                    

                    # plot preview plot
                    path = previewPath + '/' + str(currEpoch)
                    plots.plotPreview(prv_x_nn, prv_error_nn, prv_t_nn[0,:,0], prv_x_rk, prv_t_rk, path)
                    utils.saveBatch(batchPath, tBatch, icBatch)
                    

            if hyperparameters['batch_grouping'] and currEpoch % hyperparameters['batch_grouping'] == 0:
                tBatch, icBatch = ml.makeBatch(hyperparameters['t_batchsize'], hyperparameters['t_range'], hyperparameters['bc_batchsize'], hyperparameters['bc_variance'])

    return