# landcover-segmentation

DeepGlobe LandCover Satellite Image Segmentation Challenge

### Running the training script on Kaggle

**Note**: Remember to have the kaggle python package installed and set the kaggle 
API credentials in your machine, for more information check out the README of this 
[repository](https://github.com/Kaggle/kaggle-api).

After that just run the `train_kaggle.py` script.
```bash
python3 train_kaggle.py
```
### TODO:

[ ] Log gradient statistics, updated/activations ratio.
[ ] Add batchnorm layers to the Unet
[ ] Implement dice loss
[ ] Add weights to the CE loss 
[ ] Try using upsampling instead of transpose convolutions
[ ] Normalize color channels when preprocessing
[ ] Tune the optimizer
[ ] Try using 2 T4 intead of 1 P100
[ ] Half precision
[ ] Tiling
[ ] What is the best resize resolution
