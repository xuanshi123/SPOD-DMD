import numpy as np
import pydmd
import inspect
from numpy import linalg as LA
import matplotlib.pyplot as plt
#
# print(np.log(-0.99285442+0.j))

# cord = np.load('cord.npy')
L1 = np.load('L1.npy')
L2 = np.load('L2.npy')
L3 = np.load('L3.npy')
L4 = np.load('L4.npy')
# L3 = np.load('L3.npy')
# print(L.shape)
T = 1/100
N=1002
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
plt.figure(figsize=(10, 5))
plt.loglog(xf[1:N // 2], np.sum(L1[1:, :], axis=1), label='mode 1')
plt.loglog(xf[1:N // 2], np.sum(L2[1:, :], axis=1), label='mode 2')
plt.loglog(xf[1:N // 2], np.sum(L3[1:, :], axis=1), label='mode 3')
plt.loglog(xf[1:N // 2], np.sum(L4[1:, :], axis=1), label='mode 4')
# # plt.loglog(xf[1:N // 2], L3[1:, 0], label='mode 0')
# plt.loglog(xf[1:N // 2], L[1:, 1], label='mode 1')
# plt.loglog(xf[1:N // 2], L[1:, 2], color='green')
# plt.loglog(xf[1:N // 2], L[1:, 3], color='green')
# plt.loglog(xf[1:N // 2], L[1:, 4], color='green')
# plt.semilogy(L[:, 5])
plt.xlabel('frequency Hz   (St = f / 1.5)')
plt.ylabel('mode energy    $\lambda_{i}$')
plt.grid()
plt.ylim([1e3, 9e5])
plt.legend()
print(xf)
plt.axvline(x=3.3, linewidth=0.8, linestyle='dashed') # 2.3 3.2 #  2.5 3.2  # 2.8 3.4
plt.axvline(x=16.7, linewidth=0.8, linestyle='dashed')
# plt.ylim([0.01, 0.1])
plt.xlim([0.09, 50.1])
# plt.savefig('spod spectrum.png')
plt.show()
# uls2 = np.load('sja/uls3.npy')
# for i in range(cord.shape[0]):
#     if cord[i, 0] == 1 and -0.07 <= cord[i, 1] < -0.066:
#         print(i)
#
# y = uls[:, 26566]
# y2 = uls2[:, 26566]
# print(y.shape, y2.shape)
# # y = uls[:, 38900] 0.5 0.1
# # print(x.shape)
# from scipy import fft
# #
# #
# # T = 1.0 / 100.0
# # x = np.linspace(0.0, N*T, N)
# # # y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
# # yf = fft(y, 200)
# # xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
# import matplotlib.pyplot as plt
# # plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
# # plt.grid()
# # plt.xlim([-1, 10])
# # plt.show()
#
# # plt.plot(y)
# # plt.show()
# from scipy import signal
#
# y = y - np.mean(y)
# y2 = y2 - np.mean(y2)
# fs = 100
# N =1002
# f, Pxx_den = signal.welch(y, fs, nperseg=300, nfft=N)
# plt.semilogy(f, Pxx_den)
# f, Pxx_den = signal.welch(y2, fs, nperseg=300, nfft=N)
# plt.semilogy(f, Pxx_den)
# plt.axvline(x=2.5)
# plt.axvline(x=10)
# plt.axvline(x=2)
# # plt.ylim([0.5e-3, 1])
# plt.xlabel('frequency [Hz]')
# plt.ylabel('PSD [V**2/Hz]')
# plt.xlim([-1, 15])
# plt.show()
#
# T = 1.0 / 100.0
#
# yf = fft(y, n=N)
# print(yf)
# from scipy.signal import hanning
#
# w = hanning(len(y))
# ywf = fft(y * w, n=N)
# xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
# import matplotlib.pyplot as plt
#
# plt.plot(xf[1:N // 2], 2.0 / N * np.abs(yf[1:N // 2]), '-b')
# plt.plot(xf[1:N // 2], 2.0 / N * np.abs(ywf[1:N // 2]), '-r')
# plt.legend(['FFT', 'FFT w. window'])
# plt.xlim([-1, 20])
# plt.grid()
# plt.show()
#
# n = 0
# # dt = 0.01
# # t = np.linspace(0, (N-1) * dt, N)
# # x1 = np.linspace(0.5, 2.5, 401)
# # x2 = np.linspace(-0.4, 0.5, 181)
# # x1grid, x2grid = np.meshgrid(x1, x2)
# #
# # cmap = 'plasma'
# # from matplotlib import cm
# # norm = cm.colors.Normalize(vmax=0.9, vmin=-0.7)
# # for ti in uls[::10]:
# #
# #     z = np.zeros_like(x1grid, dtype=complex)
# #
# #     # zz = np.zeros_like(updated_mode[0], dtype=complex)
# #     # for mode, omg in zip(updated_mode, omega_list):
# #     #     zz += recon(omg, ti, mode)
# #     k = 0
# #
# #     for i in range(181):
# #         for j in range(401):
# #
# #             if abs(x1grid[i, j] - cord[k, 0]) < 0.0001 and abs(x2grid[i, j] - cord[k, 1]) < 0.0001:
# #                 z[i, j] = ti[k]
# #                 k += 1
# #
# #     plt.clf()
# #     plt.ylabel('y')
# #     plt.imshow(z.real, cmap=cmap, norm=norm)
# #     plt.colorbar(label='vorticity')
# #     plt.gca().invert_yaxis()
# #     plt.savefig('simulation/time {}.png'.format( t[n]* 100))
# #     n +=1
