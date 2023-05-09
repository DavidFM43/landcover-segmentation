**Deepglobe Landcover Classification Challenge 2018**

The Deepglobe Landcover Classification Challenge 2018 was one of three challenges launched by DeepGlobe in 2018 in order to push forward the research of computer vision technologies in the field of satellite images. 

## Setup

### Dataset download and preparation
The dataset from the DeepGlobe 2018 Landcover Classification challenge is hosted in Kaggle. For ease of use you can just run the python script `prepare.py` that is located in the `data` folder. This script will download the dataset using the python Kaggle API and prepare it for training. Remember that in order for this to work you need to have the Kaggle API downloaded (`pip install -r requirements.txt`) and also have your kaggle private key properly set up in your machine. After that you can run the following command from the root of the project:
```python
python3 data/prepare.py
```
This will fill the data 
```
data
├── images    -> RGB satellite images
├── masks     -> gts segmentation masks in label encoding format 
└── raw_masks -> gts segmentation masks in RGB format 
```

## Dataset
The dataset consists of a total of 803 satellite images of 2444x2444 pixels, each image comes with a segmentation mask that classifies the pixels of the image in the following classes: 
|File Name| Explanation / Function |
|---------|------------|
|Urban land| Man-made, built up areas with human artifacts|
|Agriculture land| Farms, any planned (i.e. regular) plantation, cropland, orchards, vineyards, nurseries,and ornamental horticultural areas; confined feeding operations|
|Rangeland| Any non-forest, non-farm, green land, grass|
|Forest land| Any land with at least 20% tree crown density plus clear cuts|
| Water| Rivers, oceans, lakes, wetland, ponds|
| Barren land| Mountain, rock, dessert, beach, land with no vegetation|
| Unknown| Clouds and others|

## Model
## Training
## Evaluation
## Inference
## References