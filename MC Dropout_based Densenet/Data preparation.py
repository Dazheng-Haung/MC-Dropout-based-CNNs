import numpy as np
import matplotlib.pyplot as plt


as_files = ['As.sim' + str(i)+'_matrix' + '.txt' for i in range(200)]
au_files = ['Au.sim' + str(i)+'_matrix' + '.txt' for i in range(200)]
hg_files = ['Hg.sim' + str(i)+'_matrix' + '.txt' for i in range(200)]
sb_files = ['Sb.sim' + str(i)+'_matrix' + '.txt' for i in range(200)]


stacked_data = np.zeros((200, 144, 144, 4))


for i in range(1, 200):
    as_img1 = np.loadtxt(as_files[i])
    as_img = (as_img1 - np.min(as_img1)) / (np.max(as_img1) - np.min(as_img1))
    au_img1 = np.loadtxt(au_files[i])
    au_img = (au_img1 - np.min(au_img1)) / (np.max(au_img1) - np.min(au_img1))
    sb_img1 = np.loadtxt(sb_files[i])
    sb_img = (sb_img1 - np.min(sb_img1)) / (np.max(sb_img1) - np.min(sb_img1))
    hg_img = np.loadtxt(hg_files[i])
    hg_img = (hg_img - np.min(hg_img)) / (np.max(hg_img) - np.min(hg_img))

    stacked_data[i, :, :, :, ] = np.stack((as_img, au_img, hg_img, sb_img), axis=-1)


np.save('SGS200-data.npy', stacked_data)

print(stacked_data.shape)
