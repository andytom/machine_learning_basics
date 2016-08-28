Machine Learning Recipes #6 Notes
=================================

Get and test the Tenserflow for Poets docker image

``` bash
docker run -it gcr.io/tensorflow/tensorflow:latest-devel python
```

``` python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```

Get the training data

``` bash
mkdir tf_files
cd tf_files
curl -O http://download.tensorflow.org/example_images/flower_photos.tgz
tar -xzf flower_photos.tgz
cd ..
```

Update training code

``` bash
docker run -it -v $(pwd)/tf_files:/tf_files  gcr.io/tensorflow/tensorflow:latest-devel
cd /tensorflow
git pull
```

Retrain the model

``` bash
python tensorflow/examples/image_retraining/retrain.py \
--bottleneck_dir=/tf_files/bottlenecks \
--how_many_training_steps 500 \
--model_dir=/tf_files/inception \
--output_graph=/tf_files/retrained_graph.pb \
--output_labels=/tf_files/retrained_labels.txt \
--image_dir /tf_files/flower_photos
```
*Can remove the ``how_many_training_steps`` arg for better accuracy*

Use the model

``` bash
curl -L https://goo.gl/tx3dqg > ./tf_files/label_image.py
docker run -it -v $(pwd)/tf_files:/tf_files  gcr.io/tensorflow/tensorflow:latest-devel
python /tf_files/label_image.py /tf_files/flower_photos/daisy/21652746_cc379e0eea_m.jpg
python /tf_files/label_image.py /tf_files/flower_photos/roses/2414954629_3708a1a04d.jpg 
```
