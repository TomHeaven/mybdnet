# BDNet-Pytorch (TomHeaven's Repo)
A decoupled learning scheme for training a burst denoising network for real-world denoising.

[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700154.pdf)][[Supp](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700154-supp.pdf)]

TomHeaven's repo uses his own training and test datasets for comparison in his paper.

# Training
+ Train for fixed Gaussian noise
```
sh train.sh
```
+ Train for real noise
```
nohup python3 train_BDNet_real_static.py > run.log 2>&1 &
```

Model weights will be saved in ../experiments/BDNet_train.

Manully copy generate G.pth weights to 
```
../pretrained_model/pretrained_model.pth # for real noise (poission-Gaussian)
../pretrained_model/pretrained_model_sigma5.pth  # for Gaussian noise level 5
../pretrained_model/pretrained_model_sigma15.pth # for Gaussian noise level 15
../pretrained_model/pretrained_model_sigma25.pth # for Gaussian noise level 25
```

# Test
+ Test for fixed Gaussian noise
```
sh test.sh
```
+ Test for real noise
```
python3 my_test_BDNet.py --opt_file=options/test/my_test_BDNet_gaussian.yml
```
The generated results will be saved in tb_logger/test_BDNet_train* folders.


# Data Structure
+ raw input image: The height and width of which must be multipe of 8, or it will trigger errors in net. The input raw pattern is 'rgbg' (fixed in data/video_denoise_test_dataset.py)
+ processed input image: NxCxHxW (4D), where N is number of frames, C is 4 for 'rggb' color channels extracted from Bayer images, H is height and W is width (1/2 of those for the original Bayer images).
