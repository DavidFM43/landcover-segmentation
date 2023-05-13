Recolecte algunos recursos que creo que me gustaría revisar después, pero esto no es prioritario:
- deci.ai blog section: https://deci.ai/blog/
- Neural networks training tips: https://deci.ai/blog/tricks-training-neural-networks/
- Model selection tips: https://deci.ai/blog/deep-learning-model-selection-tips/. 
- Repositorio "**[deepglobe_land_cover_classification_with_deeplabv3plus](https://github.com/GeneralLi95/deepglobe_land_cover_classification_with_deeplabv3plus)**".
- Imagenet Sota: https://deci.ai/blog/resnet50-how-to-achieve-sota-accuracy-on-imagenet/


## Papers
Para entender mas acerca del problema que estamos intentando resolver, el cual es específicamente la tarea de *Landcover Classification* con el dataset de *DeepGlobe*, decidí buscar papers en los cuales resolvieran exactamente este mismo problema, los papers que encontré son los siguientes:
La idea ahora es leer cada paper pero mas importante aún, orientar la lectura a extraer la información mas relevante para nosotros en nuestro problema y que se traduzca fácilmente en tareas accionables.
- [Ultra-high Resolution Image Segmentation via Locality-aware Context Fusion and Alternating Local Enhancement](https://paperswithcode.com/paper/from-contexts-to-locality-ultra-high)
- https://ieeexplore.ieee.org/abstract/document/10097646
- https://openaccess.thecvf.com/content_CVPRW_2020/papers/w11/Russwurm_Meta-Learning_for_Few-Shot_Land_Cover_Classification_CVPRW_2020_paper.pdf
- https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Collaborative_Global-Local_Networks_for_Memory-Efficient_Segmentation_of_Ultra-High_Resolution_Images_CVPR_2019_paper.pdf


### **Ultra-high Resolution Image Segmentation via Locality-aware Context Fusion and Alternating Local Enhancement**

**Motivación**
Hay varios factores importantes a considerar:
- Los modelos de segmentación semantica suelen ser mas exigentes en terminos de memoria de GPU.
- Las imagenes satelitates tienen una resolucion ultra-alta, lo cual añadido al punto anterior, significa un problema.
- Ante esto se plantean las soluciones basicas, que son:
	- Redimensionar las imagenes a una resolución mucho mas pequeña para estar dentro de las limitaciones de la GPU
	- Dividir la imagen original en parches de menor tamaño, predecir de forma independiente en cada parche y luego juntar las prediciones de los parches en una sola imagen.
Entiendo que lo interesante de la propuesta es que integran información global y local a nivel de parches.

- Random split en 454, 207 y 142 para entrenamiento, validación y testeo.
- Global image
- Local patches
- Focal-loss loss function
- Optimizador: Adam
- Usan la FPN(Feature Pyramid Network) con ResNet50 como backbone.


Quiero probar primero lo sencillo y luego escalar en complejidad. Por lo tanto, primero voy a probar las metodologias sencillas, es decir, resize y local patches con algunos modelos, principalmente la Unet. Quiero intentar replicar sus apreciaciones sobre el estado del arte en terminos de mIoU y GPU memory usage.

Teniendo esto en cuenta, lo primero que voy a hacer es cacharrear el repositorio que ellos tienen en github e intentar experimentar un poco con este.

