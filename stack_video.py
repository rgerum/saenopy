import imageio
import numpy as np


writer = imageio.get_writer("stack.mp4", quality=5)

for i in range(0, 500, 2):
    print(i)
    imB0 = imageio.imread(fr"\\131.188.117.96\biophysDS\dboehringer\Platte_4\Measurements_NK_TFM\single-cell-tfm-tom-paper\20170914_A172_rep1\Before\Mark_and_Find_001_Pos001_S001_z{i:03d}_ch00.tif")
    imB1 = imageio.imread(fr"\\131.188.117.96\biophysDS\dboehringer\Platte_4\Measurements_NK_TFM\single-cell-tfm-tom-paper\20170914_A172_rep1\Before\Mark_and_Find_001_Pos001_S001_z{i:03d}_ch01.tif")
    imA0 = imageio.imread(fr"\\131.188.117.96\biophysDS\dboehringer\Platte_4\Measurements_NK_TFM\single-cell-tfm-tom-paper\20170914_A172_rep1\After\Mark_and_Find_001_Pos001_S001_z{i:03d}_ch00.tif")
    imA1 = imageio.imread(fr"\\131.188.117.96\biophysDS\dboehringer\Platte_4\Measurements_NK_TFM\single-cell-tfm-tom-paper\20170914_A172_rep1\After\Mark_and_Find_001_Pos001_S001_z{i:03d}_ch01.tif")
    imB = np.hstack((imB0, imB1))
    imA = np.hstack((imA0, imA1))
    im = np.vstack((imB, imA))
    print(im.shape, im.dtype)
    im[-int(i/500*im.shape[0]):, -10:] = 0
    writer.append_data(im[::2, ::2])

writer.close()