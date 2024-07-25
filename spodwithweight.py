
import numpy as np
import os
import shutil
import scipy
import matplotlib.pyplot as plt
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
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
from scipy.signal import blackman

def SPOD(X, nDFT, nBlks, dT, weight):

    nx = X.shape[1]
    nt = X.shape[0]

    nOvlp = round(nDFT * 3 / 4) 

    window = blackman(nDFT).reshape(nDFT, 1)
    winWeight = 1 / np.mean(window)
    N = 1002

    xf = np.linspace(0.0, 1.0 / (2.0 * dT), N // 2)

    nFreq = len(xf)
    Q_hat = np.zeros((nFreq, nx, nBlks), dtype=np.csingle)

    for i in range(nBlks):
        offset = min(i*(nDFT-nOvlp)+nDFT, nt)-nDFT
        end = nDFT + offset
        Q_blk = X[offset:end] - np.mean(X[offset:end], axis=0)
        Q_blk = Q_blk * window
        Q_blk_hat = winWeight / nDFT * np.fft.rfft(Q_blk, N, axis=0)
        Q_blk_hat[1:nFreq] = 2 * Q_blk_hat[1:nFreq]
        Q_hat[:, :, i] = Q_blk_hat[:nFreq]
        print(offset, end)

    np.save('Qhat', Q_hat)

    L = np.zeros((nFreq, nBlks))
    P = np.zeros((nFreq, nx, 5), dtype=np.csingle)

    for i in range(nFreq):

        Q_hat_f = Q_hat[i, :, :]
        C = dot(Q_hat_f.conj().T, Q_hat_f * weight[:, np.newaxis]) / nBlks
        Lambda, Theta = eig(C)
        index = np.argsort(np.abs(Lambda))[::-1]
        Lambda = Lambda[index]
        Theta = Theta[:, index]
        Psi = dot(dot(Q_hat_f, Theta), np.diag(1 / np.sqrt(Lambda) / np.sqrt(nBlks)))
        P[i, :, :5] = Psi[:, :5]
        L[i, :] = np.abs(Lambda)[:]
        print(i)

    np.save('P', P)
    np.save('L', L)



if __name__ == "__main__":


    cord = np.load('cord.npy')
    cmap = 'plasma'
    dt = 0.01
    u = np.load('uls.npy')
    cord = np.load('cord.npy')
    print(u.shape, cord.shape)
    # v = np.load('ulss.npy')
    # v = v[:, -len(cord):]
    weight = np.zeros(cord.shape[0])
    for i in range(cord.shape[0]):
        if 0.4 <= cord[i, 0] <= 0.8:
            weight[i] = 1
    weight = np.concatenate((weight, weight), axis=0)

    D = u # np.concatenate((u, v), axis=1)

    D = D[-1500:, :]
    M = np.mean(D, axis=0)
    D -= M


    SPOD(D, 500, 9, 0.01, weight)
    # C = dot(D, D.T) / D.shape[0]
    #
    # lamda, W = eig(C)
    #
    # index = np.argsort(lamda)[::-1]
    #
    # lamda = lamda[index]
    # W = W[:, index]
    #
    # PHI = dot(D.T, W)
    # PHI = normalize(PHI, axis=0, norm='l2')
    # A = dot(D, PHI)
    # print(np.sum(PHI[:, 0] * PHI[:, 0]))
    #
    # plt.clf()
    # # plt.plot(A[:, 0])
    # # plt.plot(A[:, 1])
    # # print(np.mean(A[:, 0]), np.mean(A[:, 0]))
    # # plt.show()
    # energy = A * A
    # energy = np.mean(energy, axis=0)
    # plt.bar(np.arange(20), energy[:20], align='center')
    # plt.xticks(np.arange(20), np.arange(20))
    # plt.savefig('mode energy.png')
    # from scipy import fft
    # from scipy.signal import hanning
    #
    # # Number of sample points
    # # sample spacing
    # T = 1.0 / 100.0
    # y = A[:, 0]
    # N = 1002
    #
    # yf = fft(y, n=N)
    #
    # w = hanning(len(y))
    # ywf = fft(y * w, n=N)
    # xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    # import matplotlib.pyplot as plt
    # print(xf, xf.shape)
    # plt.plot(xf[1:N // 2], 2.0 / N * np.abs(yf[1:N // 2]), '-b')
    # plt.plot(xf[1:N // 2], 2.0 / N * np.abs(ywf[1:N // 2]), '-r')
    # plt.legend(['FFT', 'FFT w. window'])
    # plt.xlim([-1, 20])
    # # plt.grid()
    # # plt.show()
    #
    # x1 = np.linspace(0.45, 0.75, 301)
    # x2 = np.linspace(0.03, 0.12, 91)
    # x1grid, x2grid = np.meshgrid(x1, x2)
    #
    # l = PHI.shape[0] // 2
    #
    # for m in range(10):
    #
    #     mode = PHI[l:, m]
    #     mode2 = PHI[:l, m]
    #     z = np.zeros_like(x1grid)
    #     z2 = np.zeros_like(x1grid)
    #     k = 0
    #
    #     for i in range(91):
    #         for j in range(301):
    #
    #             if abs(x1grid[i, j] - cord[k, 0]) < 0.0001 and abs(x2grid[i, j] - cord[k, 1]) < 0.0001:
    #                 z[i, j] = mode[k]
    #                 z2[i, j] = mode2[k]
    #                 k += 1
    #
    #     if m in (0, 2):
    #
    #         fname = "mode {}".format(m)
    #
    #         with open(fname + '.dat', 'w') as f:
    #             f.write('TITLE = \"Profile\"\n')
    #             f.write('VARIABLES = \"x\", \"y\", \"u\", \"v\"\n')
    #             f.write('Zone, I={}, J={} F=BLOCK\n'.format(301, 91))
    #
    #             for data in x1grid, x2grid:
    #                 for val in data.flatten():
    #                     f.write('{}\n'.format(round(val, 5)))
    #
    #             for data in z2, z:
    #                 for val in data.flatten():
    #                     f.write('{:.5f}\n'.format(val * 1000))
    #
    #     plt.clf()
    #     plt.ylabel('y')
    #     plt.imshow(z.real, cmap=cmap)
    #     plt.colorbar(label='V')
    #     plt.gca().invert_yaxis()
    #     plt.savefig('mode {}.png'.format(m))
    #
    #     y = A[:, m]
    #     yf = fft(y, n=N)
    #
    #     w = hanning(len(y))
    #     ywf = fft(y * w, n=N)
    #     xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    #
    #     plt.clf()
    #     plt.plot(xf[1:N // 2], 2.0 / N * np.abs(yf[1:N // 2]), '-b')
    #     plt.plot(xf[1:N // 2], 2.0 / N * np.abs(ywf[1:N // 2]), '-r')
    #     plt.legend(['FFT', 'FFT w. window'])
    #     plt.xlim([-1, 20])
    #     plt.grid()
    #     plt.savefig('FFT mode {}.png'.format(m))

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
