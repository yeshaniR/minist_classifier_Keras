import numpy as np 
from keras.layers import Conv2D
from keras.regularizers import l2
from keras.regularizers import l1


noise_factor = 0.25
x_train_noisy = training_images + noise_factor + np.random.normal(loc = 0.0, scale = 1.0 , size = training_images.shape)
x_test_noisy = test_images + noise_factor + np.random.normal(loc = 0.0 , scale = 1.0, size = test_images.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)



model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1), 
                         kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')])
    
    
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x_train_noisy, training_labels, epochs=10)
test_loss = model.evaluate(test_images, test_labels)
