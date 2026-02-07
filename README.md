# Semi-CycleSAM
Semi-CycleSAM: SAM-based Cyclic Mutual Knowledge Distillation for Semi-supervised Medical Image Segmentation.

We pioneer the investigation of existing limitations in SAM-based SSMIS framework and propose Semi-CycleSAM, a novel framework providing a holistic solution to mitigate its adverse effects.

<div align="center">
  <img src="https://github.com/dhhdy/Semi-CycleSAM/blob/main/framework.png">
</div>

The code is only for research purposes. If you have any questions regarding how to use this code, feel free to contact us (hongxiangpeng2001@163.com). 

# Requirements
* Python==3.10.16
* torch==1.13.0
* torchvision==0.14.0
* numpy==1.26.4
* opencv-python
* tqdm
* scikit-image
* medpy
* pillow
* h5py
* scipy
* ```pip install -r requirements.txt```

# Acknowledgments
The code is mainly adapted from [UA-MT](https://github.com/yulequan/UA-MT), [SSL4MIS](https://github.com/HiLab-git/SSL4MIS), [SemiSAM](https://github.com/YichiZhang98/SemiSAM) and [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D). Thanks to all the authors for their contribution.
