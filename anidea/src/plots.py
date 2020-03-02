# AnIdea (c) 2019 Dominique F. Garmier All Rights Reserved
# Version: pre2.0
# --------------------------------------------------------

import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

def plotPreview(x_nn, error_nn, t_nn, x_rk4, t_rk4, path):
    '''creates preview plot of nn and saves it.
    Args:

    x_nn: array of the values of the nn at all times in t_nn
    t_nn: times at which the nn was evaluated
    error_nn: the error of the nn evaluated at all times in t_nn

    x_rk4: array of x values evaluated by any numerical method
    t_rk4: corresponding time steps

    path: path of ../instance/saves

    returns: None
    '''

    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif', serif='Times')
    plt.plot(t_nn, x_nn, c='k', lw=0.6)
    plt.plot(t_nn, error_nn, c='k', ls='--', lw=0.6)
    plt.plot(t_rk4, x_rk4, c='k', ls='dotted', lw=0.6)
    plt.legend(("NN","Er(NN)","RK4",), loc="upper right")

    ax = plt.gca()
    ax.set_xlabel('t')
    ax.set_ylabel('x(t)')

    path += '.pdf'
    plt.savefig(path, format='pdf')

    plt.clf()
    return


def plotNN(t_range, x_nn, xdot_nn, error_nn, t_nn, x_rk4, t_rk4, plots_path, data_path, toDat, toPng):
    '''creates a plot of the evaluated nn.
    Args:

    t_range: highest value in t_nn
    x_nn: array of the values of the nn at all times in t_nn
    t_nn: times at which the nn was evaluated
    error_nn: the error of the nn evaluated at all times in t_nn

    x_rk4: array of x values evaluated by any numerical method
    t_rk4: corresponding time steps

    data_path: path of ../instance/data
    plots_path: path of ../instance/plots

    toDat: bool specifing if plot is saved as .dat
    toPng: bool specifing if plot is saved as .png or displayed

    returns: None
    '''

    FileName = 'nnplot' + dt.datetime.now().strftime("%Y%m%d%H%M%S")
    pngPath = plots_path + "/" + FileName + '.pdf'
    datPathNN = data_path + "/" + FileName + '_nn.dat'
    datPathRK4 = data_path + "/" + FileName + '_rk4.dat'

    if toDat:

        f1 = open(datPathNN, 'w+')
        f1.write("x error t\n")
        
        for i in range(len(t_nn)):

            line = str(x_nn[i]) + " " + str(error_nn[i]) + " " + str(t_nn[i]) + "\n"
            f1.write(line)

        f1.close()

        f2 = open(datPathRK4, 'w+')
        f2.write("x t\n")
        
        for i in range(len(t_rk4)):

            line = str(x_rk4[i]) + " " + str(t_rk4[i]) + "\n"
            f2.write(line)

        f2.close()

    #plt.rc('text', usetex=True)
    #plt.rc('font', serif='Times')
    #plt.plot(t_nn, energyerror)
    plt.plot(t_nn, x_nn, c='k', lw=0.6)
    plt.plot(t_nn, error_nn, c='k', lw=0.6, ls='--')
    plt.plot(t_rk4, x_rk4, c='k', lw=0.6, ls='dotted')
    plt.legend(("NN","Er(NN)","RK4",), loc="upper right")
    
    ax = plt.gca()
    ax.set_xlabel('t')
    ax.set_ylabel('x(t)')

    if toPng:
        plt.savefig(pngPath, format='pdf')
    else:
        plt.show()

    plt.clf()
    return


def plotPhasespaceLoss(phasespaceLoss, v0_res, x0_res, v0_cutoff, plots_path, data_path, toDat, toPng):
    ''' creates plot of the phasespaceloss
    Args:

    phasespaceloss: [n x m] array of phasenraumloss

    v0_res: n
    x0_res: m

    v0_cutoff: max_v0
    
    plots_path: path to save plots to
    data_path: path to save .dat files to
    
    toDat: bool saying if dat is genereated
    toPng: bool saying if png is generated

    returns: None
    '''


    FileName = 'phasespaceLoss' + dt.datetime.now().strftime("%Y%m%d%H%M%S")
    pngPath = plots_path + "/" + FileName + ".pdf"
    datPath = data_path + "/" + FileName + ".dat"

    if toDat:
        
        f = open(datPath, 'w+')
        f.write("x0 v0 logloss \n")


        for i in range(np.shape(phasespaceLoss)[0]):
            for j in range(np.shape(phasespaceLoss)[1]):

                v0_pos = ((i + 0.5)/ v0_res)*v0_cutoff - v0_cutoff
                x0_pos = ((j + 0.5)/ x0_res)*np.pi

                line = str(x0_pos) + " " + str(v0_pos) + " " + str(phasespaceLoss[i,j]) + "\n"
                f.write(line)

        f.close()

    #plt.rc('text', usetex=True)
    #plt.rc('font', serif='Times')
    plt.imshow(phasespaceLoss, extent=[-np.pi, np.pi, -v0_cutoff, v0_cutoff], aspect="auto")
    plt.colorbar()

    plt.title('log Loss')

    ax = plt.gca()
    ax.set_xlabel("x0")
    ax.set_ylabel("v0")


    if toPng:
        plt.savefig(pngPath, format='pdf')
    else:
        plt.show()

    plt.clf()
    return 0


def plotLogLoss(loss, epochs, plots_path, data_path, toDat, toPng):
    '''creates plot of loss over time.
    Args:

    loss: array of loss data
    epochs: array of epochs number

    plots_path: path to save plots
    data_path: path to save .dat file

    toDat: bool saying if dat is genereated
    toPng: bool saying if png is generated

    returns: Nones
    '''

    FileName = 'lossplot' + dt.datetime.now().strftime("%Y%m%d%H%M%S")
    pngPath = plots_path + "/" + FileName + ".pdf"
    datPath = data_path + "/" + FileName + ".dat"

    def rolling_mean(x, n):

        mean = np.zeros(shape=[len(x) - (n-1)], dtype = np.float32)
        for i in range(len(x) - (n-1)):
            mean[i] = np.mean(x[i:i+n])
        return mean

    n = 2500
    if len(epochs) == 0:
        print("epochs.anidea is empty!")
        return

    meanLoss = rolling_mean(loss, n) # has length x - (n-1)
    stdevLoss = np.sqrt(rolling_mean(np.square(loss[n-1:] - meanLoss), n)) # has length x - (2n - 2)

    upperLim = meanLoss[n-1:] + stdevLoss
    lowerLim = meanLoss[n-1:] - stdevLoss

    if toDat:

        f = open(datPath, 'w+')
        f.write('epoch loss mean stdev\n')

        for lineN in range(len(loss)):


            if lineN < (2*n + 1):

                ep = epochs[lineN]
                ls = loss[lineN]
                ml = meanLoss[lineN - (n + 1)]
                st = stdevLoss[lineN - (2*n + 2)]

                f.write(str(ep) + ' ' + str(ls) + ' ' + str(ml) + ' ' + str(st) + '\n')

            elif lineN < n:

                ep = epochs[lineN]
                ls = loss[lineN]
                ml = meanLoss[lineN - (n + 1)]

                f.write(str(ep) + ' ' + str(ls) + ' ' + str(ml) + ' 0\n')

            else:
                
                ep = epochs[lineN]
                ls = loss[lineN]

                f.write(str(ep) + ' ' + str(ls) + ' 0 0\n')

        f.close()

    #plt.rc('text', usetex=True)
    #plt.rc('font', serif='Times')
    #plt.plot(epochs, np.log10(loss), linestyle="", lw=0, marker=",")
    plt.plot(epochs[n-1:],  np.log10(meanLoss), lw=0.6, c='k')
    plt.plot(epochs[2*n-2:],  np.log10(upperLim), c='k', ls='--', lw=0.6)
    #plt.plot(epochs[2*n-2:],  np.log10(lowerLim), c='k', ls='--', lw=0.2)
    plt.legend(('mean','mean + std',), loc='upper right')

    ax = plt.gca()
    ax.set_ylabel("log Loss")
    ax.set_xlabel("epochs")


    if toPng:
        plt.savefig(pngPath, format='pdf') #dpi=500)
    else:
        plt.show()

    return


def plotStepsizeScatter(h_plot, e_plot, doFit, toDat, toPng, plots_path, data_path):

    log_e = np.log10(e_plot)
    log_h = np.log10(h_plot)

    FileName = 'stepsizeplot' + dt.datetime.now().strftime("%Y%m%d%H%M%S")
    pngPath = plots_path + "/" + FileName + ".pdf"
    datPath = data_path + "/" + FileName + ".dat"

    if toDat:
        
        f = open(datPath, 'w+')
        f.write("h e \n")

        for i in range(np.shape(h_plot)[0]):

            line = str(h_plot[i]) + " " + str(e_plot[i]) + "\n"
            f.write(line)
        
        f.close()

    #plt.rc('text', usetex=True)
    #plt.rc('font', serif='Times')
    plt.plot(log_h, log_e, linestyle="", lw=0, marker="x", markersize=3, markeredgewidth = 0.3)
    plt.legend(('Stepsize Error',))

    ax = plt.gca()
    ax.set_xlabel("log h")
    ax.set_ylabel("log e(h)")

    if toPng:
        plt.savefig(pngPath, format='pdf')
    else:
        plt.show()

    return
