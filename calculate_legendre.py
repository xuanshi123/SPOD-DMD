import numpy as np
import os
import shutil
import scipy
import matplotlib.pyplot as plt
import seaborn
import random
import time as tt
from matplotlib import cm


def calculate_legendre(x):
    P = np.ones((6, len(x), 2))

    P[0] = 1
    P[1] = x
    P[2] = 0.5 * (3 * x ** 2 - 1)
    P[3] = 0.5 * (5 * x ** 3 - 3 * x)
    P[4] = (35 * x ** 4 - 30 * x ** 2 + 3) / 8
    P[5] = (63 * x ** 5 - 70 * x ** 3 + 15 * x) / 8

    leg = np.zeros((len(x), 15))
    leg[:, 0] = 1
    index = 1

    for i in range(1, 5):
        for j in range(i + 1):
            leg[:, index] = P[i - j, :, 0] * P[j, :, 1]

            index += 1

    return leg


def mean_confidence_interval(data):
    m = np.mean(data)
    lo = np.percentile(data, 2.5)
    up = np.percentile(data, 97.5)
    return m, lo, up


if __name__ == "__main__":

    a1 = np.load('ts11.npy')
    a2 = np.load('ts21.npy')
    a3 = np.load('ts31.npy')

    a11 = a1[::2, :]
    a11[1, :] = (a11[1, :] - 0.00695) / 0.0001
    a21 = a2[::2, :]
    a21[1, :] = (a21[1, :] - 0.00695) / 0.0001
    a31 = a3[::2, :]
    a31[1, :] = (a31[1, :] - 0.00695) / 0.0001
    print(a1[::2, :])
    leg1 = calculate_legendre(a11.T)
    leg2 = calculate_legendre(a21.T)
    leg3 = calculate_legendre(a31.T)

    aa = np.zeros((1001, 2))

    aa[:, 0] = np.arange(1001.0)/1000
    aa[:, 1] = 0.5
    legff = calculate_legendre(aa)
    c, res, rank, s = np.linalg.lstsq(leg1, a1[1, :], rcond=-1)

    solution1 = np.dot(legff, c)
    max_index = np.argmax(solution1)
    print(max_index/1000 *7 + 1, solution1[max_index], np.mean(solution1), np.std(solution1, ddof=1))

    c, res, rank, s = np.linalg.lstsq(leg2, a2[1, :], rcond=-1)

    solution2 = np.dot(legff, c)

    max_index = np.argmax(solution2)
    print(max_index/1000 *7 + 1, solution2[max_index], np.mean(solution2), np.std(solution2, ddof=1))
    c, res, rank, s = np.linalg.lstsq(leg3, a3[1, :], rcond=-1)

    solution3 = np.dot(legff, c)
    max_index = np.argmax(solution3)
    print(max_index/1000 *7 + 1, solution3[max_index], np.mean(solution3), np.std(solution3, ddof=1))
    plt.clf()
    norm = cm.colors.Normalize(vmax=12, vmin=1.5)
    cmap = 'plasma'
    colors = aa[:, 0] * 7 * 1.5 + 1.5
    plt.plot(colors, solution1, '-.', label='\N{GREEK SMALL LETTER THETA} = 10\N{DEGREE SIGN}')
    plt.plot(colors, solution2, '--', label='\N{GREEK SMALL LETTER THETA} = 20\N{DEGREE SIGN}')
    plt.plot(colors, solution3, label='\N{GREEK SMALL LETTER THETA} = 30\N{DEGREE SIGN}')
    plt.ylabel('Cl/Cd')
    plt.legend()
    plt.ylim([19, 25])
    plt.xlabel('frequency Hz')
    plt.savefig('scatter 1.png')
    #
    # plt.clf()
    # # norm = cm.colors.Normalize(vmax=0.4, vmin=0.2)
    # plt.ylabel('Cl/Cd')
    # plt.xlabel('frequency Hz')
    # plt.scatter(colors, solution, c=solution2, cmap=cmap, s=10)
    # plt.colorbar(label='C\N{GREEK SMALL LETTER mu}')
    # plt.savefig('scatter 2.png')
    #
    # plt.clf()
    # norm = cm.colors.Normalize(vmax=0.4, vmin=0.2)
    # plt.ylabel('C\N{GREEK SMALL LETTER mu}')
    # plt.xlabel('frequency Hz')
    # plt.scatter(colors, solution2, c=colors2, cmap=cmap, norm=norm, s=10)
    # plt.colorbar(label='amplitude m/s')
    # plt.savefig('scatter 0.png')
    #
    # plt.clf()
    # norm = cm.colors.Normalize(vmax=12, vmin=1.5)
    # plt.ylabel('C\N{GREEK SMALL LETTER mu}')
    # plt.xlabel('amplitude m/s')
    # plt.scatter(colors2, solution2, c=colors, cmap=cmap, norm=norm, s=10)
    # plt.colorbar(label='frequency  Hz')
    # plt.savefig('scatter 00.png')
    #
    # fre = []
    # mom = []
    # cl = []
    # for i in range(6000):
    #     if 0.00695 <= solution2[i] <= 0.00705:
    #         fre.append(a2[i, 1])
    #         mom.append(solution2[i])
    #         cl.append(solution[i])
    #
    # ts = np.zeros((3, len(fre)))
    # ts[0, :] = np.array(fre)
    # ts[1, :] = np.array(cl)
    # ts[2, :] = np.array(mom)
    # np.save('ts', ts)
    #
    # plt.clf()
    # # norm = cm.colors.Normalize(vmax=12, vmin=1.5)
    # plt.ylabel('C\N{GREEK SMALL LETTER mu}')
    # plt.xlabel('amplitude m/s')
    # plt.scatter(fre, cl, c=mom, cmap=cmap, s=10)
    # plt.colorbar(label='frequency  Hz')
    # plt.savefig('scatter 001.png')

    # 2.6100000000000003
    # 22.590209379105612
    # 21.96005350712671
    # 0.6370554494355128
    # 2.638
    # 23.05430182057441
    # 22.600489374570707
    # 0.4990557206161486
    # 2.855
    # 23.424790171261744
    # 23.095268881972334
    # 0.3497860083314112