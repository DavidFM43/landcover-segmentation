
## Datos

Es importante conocer muy bien los datos, pues:
- El modelo es una version comprimida del dataset, si conocemos bien el dataset podemos evaluar mejor los errores en las predicciones y dar posibles razones de porque están ocurriendo. Si la red esta dando algo que no es consistente con lo que hemos visto en los datos, algo anda mal.
Para conocer bien el dataset nos podemos hacer las siguientes preguntas:
- ¿Que factores consideramos a la hora de clasificar una imagen?
	- ¿Se necesita mucho contexto global o con la información local basta?
	- ¿Que tan importante es la resolución de las imágenes, en verdad se necesita tanta calidad de detalles para clasificar las imágenes satisfactoriamente?
	- ¿Que tanta variabilidad hay en los datos, que forma toma esta variación?
	- ¿Que variabilidad es irrelevante y se puede eliminar en el pre-procesamiento?
	- ¿Tiene importancia la posición espacial en las imágenes?
	- ¿Que tan ruidosas son las etiquetas?

#### **Insights**

- La clase de **agricultura** es fácil de reconocer por los rectángulos que formas los cultivos en la tierra, los colores son verdes-amarillos.
- La clase de **urbano** es fácil de reconocer pues son casas o edificios pequeños y aglomerados, la mayoría de veces son blancos.
- La clase de **agua** se puede reconocer por el hecho de que la textura es muy uniforme con un color cercano al azul aunque no siempre, a veces es mas amarillento o verdoso.
- La clase de **rangeland**(potreros) es mas difícil de reconocer, un factor importante es el color similar a la clase de **agricultura** pero con la ausencia de los rectángulos, a veces tiene unos pocos arboles, no demasiados y no demasiado aglomerados. En general es difícil de reconocer los limites y se puede confundir con la clase de **barrenland**. Los colores ayudan a veces pero otras veces son una distracción.
- La clase de **bosque** es complicada, pues la mayor parte del tiempo es fácil reconocer los arboles, sin embargo, no cualquier grupo de arboles se clasifica como un bosque. En la mayoría de veces se necesita que haya cierto nivel de aglomeración en los arboles, pero a veces parece que este criterio se relaja un poco.
- La clase mas difícil de clasificar en mi opinion es la de **barrenland**, lo que mas importante parece ser un color alejado del verde, pero a veces puede ser cafe.

En general me parece que las anotaciones son ambiguas, tanto en el relleno como en los bordes. En muchos casos parece que, fija una clase, las anotaciones no son consistentes. También hay muchos bordes que no son muy precisos y a veces anotaciones que no aparecen.

# **Experimentos**
### **[dazzling-terrain-58](https://wandb.ai/landcover-classification/ml-experiments/runs/998fimf1)** run: Configuración inicial

### TODO:
- [ ] ¿Que se puede decir de las imagines de las predicciones? No sé porque no he podido poner las predicciones de validación bien en el WandB xd. Seria bueno poder comparar lo que dicen las gráficas a través del tiempo con estas imágenes.
- [ ] ¿Que hacer con el *plateau* de IoU durante validación?
- [ ] ¿Que hacer con las clases "difíciles"?
- [ ] Los entrenamientos se demoran mucho, creo que seria bueno ver como optimizar esto para experimentar mas rápido, pero esto no es tan prioritario.
- [ ] ¿Que mas información se puede extraer de este primer experimento?

```
{'batch_size': 6,  
 'ce_weights': [0.8986999988555908,  
                0.4090999960899353,  
                0.9164999723434448,  
                0.8885999917984009,  
                0.9642999768257141,  
                0.9230999946594238,  
                0.0],  
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

**Insights**
**Varios**
Considerando que:
- Una epoch de entrenamiento demora 3min20sec +/- 20secs, es decir, **2.2 img/sec**.
- Una epoch de validación demora 2min30sec, es, decir, **1.4 img/sec**.
Se me hace extraño que los ciclos de validación tomen mas tiempo que los de entrenamiento, pues en validación no se guardan los gradientes en los cómputos. Esto debe ser casi seguramente por el logging de las imágenes de las predicciones en WandB
**IoU**
- El IoU medio por epoch empieza en 0.2 y termina en 0.8 para entrenamiento y **0.65** para validación. 
- El IoU medio en validación no mejora a partir de la epoch 20.
![[Pasted image 20230519154455.png]]
¿Pero que ocurre cuando vemos el IoU por clase?
- Durante el entrenamiento las clases empiezan en valores diferentes pero eventualmente llegan a un rango muy similar.
- En validación se nota que la clase verde de *rangeland* y la naranja de *barrenland* son clases difíciles de generalizar. Con respecto a esto pienso en ver las predicciones de imágenes que contengan estas dos clases para ganar mas intuición de porque pasa esto. 
	- ¿Que se puede hacer para mejorar las predicciones de clases "difíciles"?
		- ¿Hacer oversampling de estas clases?
		- ¿Darle mas peso a estas clases en especifico en la función de perdida?
![[Pasted image 20230519160903.png]]
**Función de perdida: CrossEntropy con pesos**
- Esta gráfica muestra la perdida a **nivel de batches**, por eso se ve tan oscilante. Sin embargo no sabría decir si este nivel de oscilación es aceptable, **quiero investigar al respecto**.
- No obstante la gráfica muestra un comportamiento decreciente en todo momento, al inicio con una pendiente mas pronunciada y al final un poco mas aplanada, pero al final la perdida esta sobre 0.2-0.1
![[Pasted image 20230519162012.png]]
- En validación la perdida es mucho mas constante
![[Pasted image 20230519162437.png]]
