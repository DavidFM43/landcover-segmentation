# landcover-segmentation

DeepGlobe LandCover Satellite Image Segmentation Challenge 2018 was one of three challenges launched by DeepGlobe in 2018 in order to push 
forward the research of computer vision technologies in the field of satellite images.

You can find the dataset in [Kaggle](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset).



### Running the training script on Kaggle

**Note**: Remember to have the kaggle python package installed and set the kaggle 
API credentials in your machine, for more information check out the README of this 
[repository](https://github.com/Kaggle/kaggle-api).

After that just run the `train_kaggle.py` script.
```bash
python3 train_kaggle.py
```
### TODO:

- [ ] Add weights to the CE loss 
- [ ] Log gradient statistics, updated/activations ratio.
- [ ] Add batchnorm layers.
- [ ] Find an appropiate initialization strategy for the network weights.
- [ ] Implement dice loss
- [ ] Try using upsampling instead of transpose convolutions
- [ ] Normalize color channels when preprocessing.
- [ ] Tune the optimizer.
- [ ] Half precision
- [ ] Tiling
- [ ] What is the best resize resolution
- [ ] Augmentations
- [ ] Try using 2 T4 intead of 1 P100.

### Questions

- [x] ¿What are the key difference between per pixel accuracy and IoU?
- [ ] ¿Why is Dice loss better than standard cross_entropy?