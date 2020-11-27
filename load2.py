import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib import cm

#
# print(np.log(-0.99285442+0.j))

# cord = np.load('cord.npy')
P = np.load('P.npy')
L = np.load('L.npy')

print(P.shape)

#x1 = np.linspace(0.4, 1.05, 651)
#x2 = np.linspace(0.0, 0.21, 211)
x1 = np.linspace(0.96, 2.5, 309)
x2 = np.linspace(-0.2, 0.5, 141)

x1grid, x2grid = np.meshgrid(x1, x2)

cord = np.load('cord.npy')
cmap = 'plasma'

ff = 58
p_f = P[ff]
mag = np.sqrt(L[ff, 0])
mode = p_f[:, 0].real * mag
mode2 = p_f[:, 0].imag * mag
z = np.zeros_like(x1grid, dtype=np.double)
z2 = np.zeros_like(x1grid, dtype=np.double)
k = 0

for i in range(len(x2)):
    for j in range(len(x1)):

        if abs(x1grid[i, j] - cord[k, 0]) < 0.0001 and abs(x2grid[i, j] - cord[k, 1]) < 0.0001:
            z[i, j] = mode[k]
            z2[i, j] = mode2[k]
            k += 1

norm = cm.colors.Normalize(vmax=0.01, vmin=-0.01)
plt.clf()
plt.ylabel('y')
plt.imshow(z.real, cmap=cmap, norm=norm)
plt.colorbar(label='V')
plt.gca().invert_yaxis()
plt.savefig('mode v.png')

fname = "mode 0 {}".format(ff)
with open(fname + '.dat', 'w') as f:
    f.write('TITLE = \"Profile\"\n')
    f.write('VARIABLES = \"x\", \"y\", \"omega_real\", \"omega_imag\"\n')
    f.write('Zone, I={}, J={} F=BLOCK\n'.format(len(x1), len(x2)))

    for data in x1grid, x2grid:
        for val in data.flatten():
            f.write('{}\n'.format(round(val, 5)))

    for data in z, z2:
        for val in data.flatten():
            f.write('{:.5f}\n'.format(val))
