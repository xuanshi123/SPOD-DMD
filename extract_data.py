import numpy as np
import os
import scipy
import matplotlib.pyplot as plt
import random
import time as tt
from matplotlib import cm
from scipy.interpolate import interp2d

if __name__ == "__main__":

    value_list = []
    value_list2 = []
    value_list3 = []
    cord = []

    ls = os.listdir(path='./')
    ls.sort()
    m = 0
    for i in ls:
        print(i)

        if os.path.isfile('./'+i):
            continue

        if float(i) < 14 or float(i) > 25:
            continue

        dirName = './' + i + '/data_U.xy'

        if float(i) >= 10:
            dirName = './' + i + '/data_U_vorticity.xy'

        print(dirName)
        f = open(dirName, "r")
        words = []
        # for b in range(2):
        #     line = f.readline()
        numberold=-99999
        while True:
            line = f.readline()
            if not line:
                break

            # process line

            words = line.split()
            numbers = [float(w) for w in words]
            if numberold !=numbers[0] and numbers[0] >= 0.96  and numbers[1] >= -0.2:
                value_list.append(numbers[3])
                value_list2.append(numbers[4])
                value_list3.append(numbers[8])
                if m == 0:
                    cord.append([numbers[0], numbers[1]])
                numberold = numbers[0]
            if numberold == numbers[0] :
                 value_list[-1] =  (value_list[-1] + numbers[3]) / 2
                 value_list2[-1] =  (value_list2[-1] + numbers[4]) / 2
                 value_list3[-1] =  (value_list3[-1] + numbers[8]) / 2

        m += 1



    cmap = 'plasma'
    norm = cm.colors.Normalize(vmax=100, vmin=-100)
    uls = np.array(value_list)
    uls2 = np.array(value_list2)
    uls3 = np.array(value_list3)
    cord = np.array(cord)

    # x1 = np.linspace(0.5, 1.6, 221)
    # x2 = np.linspace(0, 0.3, 61)
    # x1grid, x2grid = np.meshgrid(x1, x2)

    # z = np.zeros_like(x1grid)
    #
    # k = 0
    # for i in range(61):
    #     for j in range(221):
    #         if abs(x1grid[i, j] - cord[k, 0]) < 0.000001 and abs(x2grid[i, j] - cord[k, 1]) < 0.000001:
    #             z[i, j] = uls[k]
    #             k +=1
    #
    # # f = interp2d(cord[:, 0], cord[:, 1], uls, kind='linear')
    # # z = f(x1, x2)
    # # print(uls.shape, z.shape)
    #
    # plt.clf()
    # plt.ylabel('y')
    # plt.xlabel('x')
    # plt.imshow(z, cmap=cmap,norm=norm)
    # plt.colorbar(label='vorticity')
    # plt.gca().invert_yaxis()
    # plt.savefig('2eeeeee.png'.format(i))


    np.save('cord', cord)
    print(cord.shape)

    uls = np.reshape(uls, (m, -1))
    np.save('uls2', uls)

    uls2 = np.reshape(uls2, (m, -1))
    np.save('uls3', uls2)

    # uls4 = np.load('uls.npy')

    uls3 = np.reshape(uls3, (m, -1))
    # uls3 = np.concatenate((uls4[-900:] , uls3 ), axis=0)
    np.save('uls', uls3)


    # sol = uls[-1]
    # uls = uls[:-1].T
    #
    # c, res, rank, s = np.linalg.lstsq(uls, sol, rcond=-1)
    # print(c)
    # np.save('c', c)