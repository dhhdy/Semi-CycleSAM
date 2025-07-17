import numpy as np
from glob import glob
from tqdm import tqdm
import h5py
import nrrd
import SimpleITK as sitk
import os
# output_size =[112, 112, 80]  # H W D
output_size =[120, 160, 160] # D H W
def covert_h5():
    listt = glob("/data1/data/lung/I_IIIA_LUAD/GD/imagesTr/*.nii.gz")
    # print(os.path.exists("/data1/data/lung/I_IIIA_LUAD/GD/imagesTr/10726547_0000.nii.gz"))
    for item in tqdm(listt):
        ed = item.split('/')[-1].split('.')[0]
        fr = ed.split('_')[0]
        print(item)
        image = sitk.GetArrayFromImage(sitk.ReadImage(item))
        label = sitk.GetArrayFromImage(sitk.ReadImage(item.replace(ed, fr).replace("imagesTr", "labelsTr")))
        label = (label == 1).astype(np.uint8)
        d, h, w = label.shape

        tempL = np.nonzero(label)
        minx, maxx = np.min(tempL[0]), np.max(tempL[0])
        miny, maxy = np.min(tempL[1]), np.max(tempL[1])
        minz, maxz = np.min(tempL[2]), np.max(tempL[2])

        px = max(output_size[0] - (maxx - minx), 0) // 2
        py = max(output_size[1] - (maxy - miny), 0) // 2
        pz = max(output_size[2] - (maxz - minz), 0) // 2
        minx = max(minx - np.random.randint(5, 10) - px, 0)
        maxx = min(maxx + np.random.randint(5, 10) + px, d)
        miny = max(miny - np.random.randint(10, 20) - py, 0)
        maxy = min(maxy + np.random.randint(10, 20) + py, h)
        minz = max(minz - np.random.randint(10, 20) - pz, 0)
        maxz = min(maxz + np.random.randint(10, 20) + pz, w)

        image = (image - np.mean(image)) / np.std(image)
        image = image.astype(np.float32)
        image = image[minx:maxx, miny:maxy, minz:maxz]
        label = label[minx:maxx, miny:maxy, minz:maxz]
        H, W, D = image.shape
        if H == 0 or W == 0 or D == 0:
            continue
        print(label.shape)
        item = item.replace('GD', 'GD_preprocess')
        f = h5py.File(item.replace('nii.gz', 'mri_norm2.h5'), 'w')
        f.create_dataset('image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        f.close()

if __name__ == '__main__':
    covert_h5()