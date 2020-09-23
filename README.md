# Rehoboam

Red neuronal para detección de tráfico.

## Clases

- Coche
    - Frontal
    - Trasero
    - Lateral izquierda
    - Lateral derecha
- Camion
    - Frontal
    - Trasero
    - Lateral izquierda
    - Lateral derecha
- Moto
    - Frontal
    - Trasero
    - Lateral izquierda
    - Lateral derecha
- Autobus
    - Frontal
    - Trasero
    - Lateral izquierda
    - Lateral derecha

## Implementación

### 15/07/2020

Las instrucciones para el reentreno de la red estan en el repositorio de [Darknet](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects)

### 16/07/2020

La implementacion con Tensorflow es la escogida a esta fecha. Consta de dos archivos:

- save_model.py -> transforma el modelo .cfg y los pesos de Yolo en un modelo de TF.
- detect.py -> usa el modelo tf y detecta las clases de la imagen.
Se ha creado un fichero tf_utils.py donde incluir las dos funciones comnetadas anteriormente para que sean importadas al programa principal.

### 29/07/2020

Las funciones se han pasado a un archivo independiente (utils.py) por limpieza.
Para hacer pruebas es ncesario levantar el cluster de Kafka con las imagenes ya configuradas.

```bash
sudo docker-compose up
```

Crear un cluster (añadir zookeper) y crear un topic. Varios errores hacen necesario eliminar todo despues.

```bash
sudo docker-compose rm -f
sudo docker-compose pull
```

Las imagenes se descargan haciendo uso de la libreria requests y se envian por un productor de Kafka (respeta el orden de los mensajes). Por otro lado, un consumidor recibe la imagen y el modelo de Tensorflow predice la imagen (14.5s CPU).
Posibles mejoras:

- Sacar la carga del modelo de la funcion utils.detect (0.3s CPU)
- Probar con batches de imagenes
- Probar en GPU

La funcion de inferencia devuelve la clase detectada y el score, se convierte a formato JSON y se envia por otro topic de Kafka para que sea leido en el Dashboard.

### 30/07/2020

Se ha comenzado con la creacion del Dashboard (dashboard.py) que será el programa principal del proyecto. Está compuesto por una barra de navegación con botones a diferentes páginas, dos selectores para el distrito y el identificador de la cámara que se desea ver y la imagen ya clasificada.

El selector de distritos carga sus opciones a partir de un Dataframe leyendo el archivo cameras_info.csv. Según el distrito seleccionado se filtra dicho Dataframe y se cargan las cámaras que pertenecen a dicho distrito. A continuación comienza el proceso de inferencia, activando el productor de kafka y el consumidor y cada 5 segundos se recarga la imagen ya clasificada.

Cada vez que se cambia de cámara se crea un nuevo tópico para Kafka con el fin de tener varios buffers de mensajes. <span style="color:red">Problemas al cambiar de cámara. Probar a realizar la carga del modelo tf antes</span>

### 07/08/2020

Se ha creado un fichero a parte (test/graph.py) para testear el dahsboard y la conexión con Mongo. En este dashboard temporal se ha creado un gráfico de barras para el conteo de clases (agregación de vehiculos? boton?) y un componente de carga para las imágenes.

### 12/08/2020

Se ha copiado el código de los ficheros de test en los ficheros principales y comentado el código. Modificación no definitiva del layout y se han añadido dos botones a la barra de navegación: uno para ver el tráfico en tiempo real (default) y otro para ver una gráfica con un histórtico de los datos. En el histórico se añadiran opciones para filtrar por la cámara y fecha deseada.

### 14/08/2020

Uno de los botones de navegación funciona conrrectamente, todavía hace falta terminar el layout de la búsqueda de datos históricos. Problemas con las fechas proporcionadas por el componente de Dash y Mongo.

Layout acabado. Faltan algunos detalles y crear las otras páginas de navegación.

### 15/08/2020

Añadido un slider (hacer más pequeño y ponerlo arriba deshabilitado) para filtrar la gráfica histórica por hora. Hace falta añadir los saltos entre horas y sólo pintar el tramo donde haya datos.

### 16/08/2020

Se ha desechado la idea de la lista con todos los segundos del día ya que no es eficiente y hay saltos entre los diferentes puntos.
Incluida la pestaña de localización de las cámaras.

### 18/08/2020

Se ha añadido una descripción de los diferentes menús y el modo de empleo y se ha modificado el estilo aunque no está terminado. Queda crear un fichero único con todos los datos relevantes y cambiar la leyenda del gráfico de barras.

### 20/08/2020

Terminado el diseño de la aplicación y copiado todo al archivo original 'dashboard.py'. Creado un archivo excel con todos los datos de las cámaras y las clases. Cuando se entrene la red se podría añadir un spinner para la carga de las imágenes.

### 21/08/2020

Terminado el estilo del dashboard y modificada la creación de los topics de Kafka, ahora comprueba si existe previamente. También se han finalizado todas las páginas de la aplicación.

### 23/08/2020

Creado un código para eliminar las imágenes que no contengan ninguna clase. También crea un gráfico de barras con la distribución de las clases (no balanceadas?)

### 26/08/2020

Terminado el etiquetado de imágenes para el reentreno (1489 archivos). Desbalanceo de las clases, buscar solución.

Para entrenar YOLO con Darknet:

```bash
./darknet detector train build/darknet/x64/data/obj.data yolov4-obj.cfg yolov4.conv.137 -map
```

Para validar YOLO con Darknet:

```bash
./darknet detector test build/darknet/x64/data/obj.data yolov4-obj.cfg yolo-obj_8000.weights
```

Entreno con CPU lento, probar con Amazon EC2 o Google Colab.

- [Google colab](https://colab.research.google.com/drive/1_GdoqCJWXsChrOiY8sZMr_zbr_fH-0Fg?usp=sharing#scrollTo=POozxsvFdXTu) -> entreno con GPU 'rápido', mAP@0.5 = ~70% (2h). Dejar 24 horas tras balanceo de clases.

### 27/08/2020

Para balancear las clases se ha optado por una mezacl de oversampling y subsampling hasta obtener el número de clases deseadas. Se han aplicado técnicas de data augmentation con la libreria [Albumentations](https://albumentations.ai/) para realizar pequeños cambios en las imágenes y realizar el oversampling.

Unificar archivos para el balance y el conteo.

### 28/08/2020

Unificado el código para el preprocessamiento de las imágenes y archivos. Se han transformado los candidatos para el oversampling y se han obtenido dos distribuciones de clases. Se probará en Google Colab si existe alguna mejora entre los diferentes datasets. Probar también con subsampling.

### 02/09/2020

Se ha creado un código para el balanceo de clases. Hacer algunos test haciendo menos copias.

### 03/09/2020

Se ha modificado el código de balanceo y se ha parametrizado para experimentar con varios escenarios. Se ha llegado a la conclusión de que el límite de balanceo debe ser 3000, la clase que finaliza el proceso de rotación debe ser la 14, con un límite de copias de 60 y la diferencia entre clases principales y las demás debe ser 0. Se siguen realizando 65 transformaciones (augmnetation) por imagen.

### 04/09/2020

Descargar nuevas imágenes de otras cámaras para el testeo de la red con un balanceo de clases de 200. El código crea los ficheros XML y también los redistribuye en dos carpetas.

### 07/09/2020

Se ha entrenado YOLO con las imágenes balanceadas y en validación hay mucha mejora. Se está creando un código para hacer una matriz de confusión y saber si mejora realmente con imágenes no vistas anteriormente.
Se ha modificado también unos detalles del dashboard. Probar con un único topic de Kafka.

### 14/09/2020

Encontrado un error en la carga del histórico de los datos. El error aparece únicamnete cuando se selcciona la fecha del 6 de Septiembre, pero tampoco muestra la información del 14 de Septiembre. Puede ser un problema de la escritura en base de datos o propio del código.
Por otro lado, no se ha podido configurar el entranamiento de TensorFlow en Google ya que salta un error al ejecutar un el código en Python que genera los .record para tf.

### 15/09/2020

Cambiada la configuración de Kafka para crear un único topic de entrada de imágenes con 2 particiones y con factor de réplica 1. Funciona bien con un único tópico pero al seleccionar una cámara después de otra empiezan a aparecer imágenes de las dos cámaras.
Revisionar los videos de Cristian sobre Kafka y mirar los ejercicios.
El error en el histórico se puede deber a los posibles datos vacios a ciertas horas.

### 16/09/2020

Se ha modificado el modo de ejecución del proyecto. Ahora se ha creado un archivo independiente que hace de consumidor de Kafka y cada 5 segundos se va llamando al productor desde la aplicación Dash. El único inconveniente es que no es posible pasar el identificador de la cámara para incorporarlo a a la base de datos. Se está estudiando Apache Thrift para serializar los diferentes parámetros.

### 17/09/2020

Funcionamiento del análisis en tiempo real terminado serializando con Avro. Solventado el error del gráfico de barras en el histórico de los datos. Falta cambiar la franja horaria a la correcta y revisar el número de horas.

### 18/09/2020

Se ha corregido el gráfico del histórico de datos para que interpole entre los valores que no se recogen. También se ha modificado el fichero yml para la creación de dos brokers de Kafka. Falta testear la franja horaria a la que se guardan los datos y el fichero sh para ejecutar todo de una sola vez.

### 23/09/2020

Terminado el dashboard y testeado con la inclusión de un spinner para la carga del gráfico histórico. Lo próximo es borrar los archivos pesados del git y hacer un código para el testeo de las diferentes redes.
