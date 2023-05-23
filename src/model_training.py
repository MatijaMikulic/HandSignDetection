from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(400, 400, 3))

#adding layer to network
myLayers = base_model.output
myLayers = GlobalAveragePooling2D()(myLayers)
myLayers = Dense(256, activation='relu')(myLayers)
predictions = Dense(3, activation='softmax')(myLayers)

model = Model(inputs=base_model.input, outputs=predictions)
              
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_dir = 'dataset/Train/'
test_dir = 'dataset/Test/'
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(400, 400), batch_size=batch_size, class_mode='categorical')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(400, 400), batch_size=batch_size, class_mode='categorical')

epochs = 2
model.fit(train_generator, epochs=epochs, validation_data=test_generator)

test_loss, test_accuracy = model.evaluate(test_generator)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)

model.save("cnn_model")