# landcover-segmentation

DeepGlobe LandCover Satellite Image Segmentation Challenge 2018 was one of three challenges launched by DeepGlobe in 2018 in order to push forward the research of computer vision technologies in the field of satellite images.  You can find the dataset in [Kaggle](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset).

The task was to classify the pixels of a satellite image in RGB format into 6 classes, namely: 

|File Name| Explanation / Function |
|---------|------------|
|Urban land| Man-made, built up areas with human artifacts|
|Agriculture land| Farms, any planned (i.e. regular) plantation, cropland, orchards, vineyards, nurseries,and ornamental horticultural areas; confined feeding operations|
|Rangeland| Any non-forest, non-farm, green land, grass|
|Forest land| Any land with at least 20% tree crown density plus clear cuts|
| Water| Rivers, oceans, lakes, wetland, ponds|
| Barren land| Mountain, rock, dessert, beach, land with no vegetation|
| Unknown| Clouds and others|


### Running the training script on Kaggle

**Note**: Remember to have the kaggle python package installed and set the kaggle 
API credentials in your machine, for more information check out the README of this 
[repository](https://github.com/Kaggle/kaggle-api).

After that just run the `train_kaggle.py` script.
```bash
python3 train_kaggle.py
```
### TODO:

- [x] Try adding weights to the CE loss.
- [x] Find an appropiate initialization strategy for the network weights.
- [ ] Implement dice loss.
- [x] Try adding batchnorm layers.
- [ ] Log gradient statistics, updated/activations ratio.
- [ ] Try using upsampling instead of transpose convolutions.
- [ ] Normalize color channels when preprocessing.
- [ ] Try other popular segmentation models from this [library](https://github.com/qubvel/segmentation_models.pytorch) 
- [ ] Try using two Tesla T4 intead of one Tesla P100.
- [ ] Implement Tiling.
- [ ] Try Augmentations.
- [ ] Half precision.
- [ ] Tune the optimizer, batch size, .
- [ ] Add **Model** section to README.
- [ ] Add **Logging** section to README.
- [ ] Add **File structure** section to README.
- [ ] Add **Dataset** section to README.
- [ ] Add **Contribute** section to README.
- [ ] Add **Setup** section to README and `requiremets.txt` file.


### Questions

- [x] ¿What are the key difference between per pixel accuracy and IoU?
- [ ] ¿Why is Dice loss better than standard cross_entropy?
- [ ] ¿What is the best resize resolution?
