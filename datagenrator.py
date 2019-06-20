from keras.preprocessing.image import ImageDataGenerator


gen_data = ImageDataGenerator(vertical_flip=True, horizontal_flip=True, shear_range=0.5#, zoom_range=0.5#, zca_whitening=True,
                              )

for images in gen_data.flow_from_directory('./data/train/', [1920,1920], 'rgb', save_to_dir='./data/geneator/', save_format='jpg'):
    images = images

    