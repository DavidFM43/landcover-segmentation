### Baseline

**Datos**:
- Para los datos no hicimos nada en especial, simplemente decidimos hacer un resize de las imagenes a un resolución mucho mas pequeña, inicialmente propusimos 512x512 pixeles.
**Modelo**:
- Unet igual al paper original
- **Weighted CE** para lidiar con el imbalance de clases, los pesos son las proporciones de cada clase en el dataset de entrenamiento.
- **Kaimming weight initialization scheme**
- Adam optimizer con learning rate de 3e-4
**Entrenamiento**
- Epochs: 5-10-50-100
- Batch size: 5-8

**Preguntas**
- ¿Que encoder usar?