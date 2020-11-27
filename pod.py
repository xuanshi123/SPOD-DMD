
import numpy as np
import os
import shutil
import scipy
import matplotlib.pyplot as plt
import seaborn
import random
import time as tt
from matplotlib import cm
import scipy
import scipy.integrate
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import dot, multiply, diag
from numpy.linalg import inv, eig, pinv, norm, solve, cholesky
from scipy.linalg import svd, svdvals
from scipy.sparse import csc_matrix as sparse
from scipy.sparse import vstack as spvstack
from scipy.sparse import hstack as sphstack
from scipy.sparse.linalg import spsolve

from matplotlib import animation
from IPython.display import HTML
from matplotlib import pyplot as plt
from past.utils import old_div
from numpy import linalg as LA
from sklearn.preprocessing import normalize
from scipy import signal

def dmd(X, Y, truncate=None):
    Q, R = np.linalg.qr(X)

    S = dot(dot(inv(R), Q.conj().T), Y)

    lamda, W = eig(S)

    print(S.shape, lamda.shape)

    T = np.vander(lamda, X.shape[1], True)
    # for i in range(T.shape[0]):
    #     T[i] /= lamda[i]

    Phi = dot(Y, inv(T))
    # r = len(Sig2) if truncate is None else truncate # rank truncation
    # U = U2[:,:r]
    # Sig = diag(Sig2)[:r,:r]
    # V = Vh2.conj().T[:,:r]
    # Atil = dot(dot(dot(U.conj().T, Y), V), inv(Sig)) # build A tilde
    # mu,W = eig(Atil)
    # Phi = dot(dot(dot(Y, V), inv(Sig)), W) # build DMD modes
    return lamda, Phi


def recon(omega, t, mode):

    amp = 1
    return (np.exp(omega * t) * mode + np.exp(omega.conj() * t) * mode.conj()) * amp

def plot_eigs(eigs,
              show_axes=True,
              show_unit_circle=True,
              figsize=(8, 8),
              title=''):
    """
    Plot the eigenvalues.

    :param bool show_axes: if True, the axes will be showed in the plot.
        Default is True.
    :param bool show_unit_circle: if True, the circle with unitary radius
        and center in the origin will be showed. Default is True.
    :param tuple(int,int) figsize: tuple in inches defining the figure
        size. Default is (8, 8).
    :param str title: title of the plot.
    """
    if eigs is None:
        raise ValueError('The eigenvalues have not been computed.'
                         'You have to perform the fit method.')

    plt.figure(figsize=figsize)
    plt.title(title)
    plt.gcf()
    ax = plt.gca()

    points, = ax.plot(
        eigs.real, eigs.imag, 'bo', label='Eigenvalues')

    # set limits for axis
    limit = 2
    ax.set_xlim((-limit, limit))
    ax.set_ylim((-limit, limit))

    plt.ylabel('Imaginary part')
    plt.xlabel('Real part')

    if show_unit_circle:
        unit_circle = plt.Circle(
            (0., 0.),
            1.,
            color='green',
            fill=False,
            label='Unit circle',
            linestyle='--')
        ax.add_artist(unit_circle)

    # Dashed grid
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    for line in gridlines:
        line.set_linestyle('-.')
    ax.grid(True)

    ax.set_aspect('equal')

    # x and y axes
    if show_axes:
        ax.annotate(
            '',
            xy=(np.max([limit * 0.8, 1.]), 0.),
            xytext=(np.min([-limit * 0.8, -1.]), 0.),
            arrowprops=dict(arrowstyle="->"))
        ax.annotate(
            '',
            xy=(0., np.max([limit * 0.8, 1.])),
            xytext=(0., np.min([-limit * 0.8, -1.])),
            arrowprops=dict(arrowstyle="->"))

    # legend
    if show_unit_circle:
        ax.add_artist(
            plt.legend(
                [points, unit_circle], ['Eigenvalues', 'Unit circle'],
                loc=1))
    else:
        ax.add_artist(plt.legend([points], ['Eigenvalues'], loc=1))

    plt.show()

if __name__ == "__main__":


    cord = np.load('cord.npy')
    cmap = 'plasma'
    dt = 0.01

    cord = np.load('cord.npy')
    u = np.load('uls.npy')
    D = u[-1000:]
    # D = np.concatenate((u, v), axis=1)

    print(D.shape)
    D = D[:]
    M = np.mean(D, axis=0)
    print(M.shape)
    for i in range(D.shape[0]):
        D[i] -= M


    C = dot(D, D.T) / D.shape[0]

    lamda, W = eig(C)

    index = np.argsort(lamda)[::-1]

    lamda = lamda[index]
    W = W[:, index]

    PHI = dot(D.T, W)
    PHI = normalize(PHI, axis=0, norm='l2')
    A = dot(D, PHI)
    print(np.sum(PHI[:, 0] * PHI[:, 0]))

    plt.clf()
    # plt.plot(A[:, 0])
    # plt.show()
    energy = A * A
    energy = lamda 
    print(lamda[0], lamda[1])
    plt.bar(np.arange(20), energy[:20], align='center')
    plt.xticks(np.arange(20), np.arange(20))
    plt.savefig('mode energy.png')
    from scipy import fft
    from scipy.signal import hamming

    # Number of sample points
    N = 1002
    # sample spacing
    T = 1.0 / 100.0
    y = A[:, 0]


    yf = fft(y, n=N)

    w = hamming(len(y))
    ywf = fft(y * w, n=N)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    import matplotlib.pyplot as plt
    print(xf, xf.shape)
    # plt.plot(xf[1:N // 2], 2.0 / N * np.abs(yf[1:N // 2]), '-b')
    plt.plot(xf[1:N // 2], 2.0 / N * np.abs(ywf[1:N // 2]), '-r')
    plt.legend(['FFT w. window'])
    plt.xlim([1, 5])
    plt.grid()
    # plt.show()

    x1 = np.linspace(0.96, 2.5, 309)
    x2 = np.linspace(-0.2, 0.5, 141)
    x1grid, x2grid = np.meshgrid(x1, x2)

    l = PHI.shape[0] // 2

    for m in range(10):

        mode = PHI[:, m]
        # mode2 = PHI[:l, m]
        z = np.zeros_like(x1grid)
        z2 = np.zeros_like(x1grid)
        k = 0

        for i in range(141):
            for j in range(309):

                if abs(x1grid[i, j] - cord[k, 0]) < 0.0001 and abs(x2grid[i, j] - cord[k, 1]) < 0.0001:
                    z[i, j] = mode[k]
                    # z2[i, j] = mode2[k]
                    k += 1

        if m < 3:

            fname = "mode {}".format(m)

            with open(fname + '.dat', 'w') as f:
                f.write('TITLE = \"Profile\"\n')
                f.write('VARIABLES = \"x\", \"y\", \"omega\",\n')
                f.write('Zone, I={}, J={} F=BLOCK\n'.format(309, 141))

                for data in x1grid, x2grid:
                    for val in data.flatten():
                        f.write('{}\n'.format(round(val, 5)))

                for data in z:
                    for val in data.flatten():
                        f.write('{:.5f}\n'.format(val * np.sqrt(lamda[m])))

        # plt.clf()
        # plt.ylabel('y')
        # plt.imshow(z.real, cmap=cmap)
        # plt.colorbar(label='V')
        # plt.gca().invert_yaxis()
        # plt.savefig('mode {}.png'.format(m))



        y = A[:, m]
        yf = fft(y, n=N)
        f, Pxx_den = signal.welch(y, 100, nperseg=333, nfft=N,  window='hamming')
        w = hamming(len(y))
        ywf = fft(y * w, n=N)
        xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)

        plt.clf()
        plt.figure(figsize=(5, 2.5))
        # plt.semilogy(f, Pxx_den)
        # plt.plot(xf[1:N // 2], 2.0 / N * np.abs(yf[1:N // 2]), '-b')
        plt.plot(xf[1:N // 2], 2.0 / N * np.abs(ywf[1:N // 2]), '-r')
        # plt.legend(['FFT hamming'])
        plt.xlabel('frequency Hz   (St = f / 1.5Hz)')
        plt.xticks(np.arange(10), np.arange(10))
        plt.yticks(rotation=90)
        plt.xlim([-0.1, 6])
        plt.ylim([-0.1, 150])

        plt.tight_layout()
        plt.grid()
        plt.savefig('SJA FFT mode {}.png'.format(m))
        # if np.abs(mu[m]) < 0.95:
        #     continue

        # if LA.norm(mode) > 1500 or (abs(freq[m]) > 14 and LA.norm(mode) > 500):
        #     print(m, abs(freq[m]))
        #
        #     z = np.zeros_like(x1grid, dtype=complex)
        #     k = 0
        #
        #     for i in range(181):
        #         for j in range(401):
        #
        #             if abs(x1grid[i, j] - cord[k, 0]) < 0.0001 and abs(x2grid[i, j] - cord[k, 1]) < 0.0001:
        #                 z[i, j] = mode[k]
        #                 k += 1
        #
        #
        #     plt.clf()
        #     plt.ylabel('y')
        #     plt.imshow(z.real, cmap=cmap, norm=norm)
        #     plt.colorbar(label='vorticity')
        #     plt.gca().invert_yaxis()
        #     plt.savefig('mode {} {}.png'.format(freq[m], m))
    #
    #         # if mu[m].imag > 0:
    #         #     for ti in t[::10]:
    #         #         print(mu[m])
    #         #         zz = recon(omega[m], ti, z)
    #         #         plt.clf()
    #         #         plt.ylabel('y')
    #         #         plt.imshow(zz.real, cmap=cmap, norm=norm)
    #         #         plt.colorbar(label='vorticity')
    #         #         plt.gca().invert_yaxis()
    #         #         plt.savefig('mode {} {}.png'.format(m, ti* 100))
    #         #
    #
