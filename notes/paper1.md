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

