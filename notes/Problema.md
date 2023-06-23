# **Experiments**

## **[dazzling-terrain-58](https://wandb.ai/landcover-classification/ml-experiments/runs/998fimf1)** run: Configuración inicial

```
{'batch_size': 6,  
 'ce_weights': [0.89, 0.40, 0.91, 0.88, 0.96, 0.92, 0.0],  
 'downsize_res': 512,  
 'epochs': 40,  
 'loss_fn': 'CrossEntropyLoss',  
 'lr': 0.0003,  
 'model_architecture': 'Unet',  
 'model_config': {'classes': 7,  
                  'encoder_name': 'resnet34',  
                  'encoder_weights': 'imagenet',  
                  'in_channels': 3},  
 'num_workers': 2,  
 'optimizer': 'Adam'}
```

### **Insights**
---
- Validation epochs are taking longer because of the image logging.
	- One training epoch takes 3 minutes and 20 seconds +/- 20 seconds, which is equivalent to **2.2 images/second**.
	- One validation epoch takes 2 minutes and 30 seconds, which is equivalent to **1.4 images/second**.
![[Pasted image 20230519154455.png]]
- Training mean IoU: 0.8
- Validation mean IoU: 0.646
![[Pasted image 20230519160903.png]]
- During training, the classes start with different values but eventually reach a very similar range.
- In validation, it is noticeable that the green class of _rangeland_ and the orange class of _barrenland_ are difficult classes to generalize.
	- Regarding this, I am thinking of looking at the predictions of images that contain these two classes to gain more insight into why this is happening.
	- What can be done to improve predictions for "difficult" classes?
		- Should oversampling be done for these classes?
		- Should more weight be given to these specific classes in the loss function?
  
![[Pasted image 20230519162012.png]]
- This graph shows the loss at the **batch level**, which is why it appears so oscillating. However, I couldn't say whether this level of oscillation is acceptable; **I want to investigate further**.
- Nevertheless, the graph shows a decreasing trend throughout, initially with a steeper slope and gradually flattening towards the end. However, the loss remains around 0.2-0.1 at the end.
