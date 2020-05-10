''' INTRO

Name: Scott Qualkenbush

Course: CIS-2532-NET01

Date: 5-10-2020

Program Summary: 

    Deep Learning exploration, Part 2: Build your first Convolutional Neural Network to recognize images

    Code credit goes to Joseph Lee Wei En

    Source: https://medium.com/intuitive-deep-learning/build-your-first-convolutional-neural-network-to-recognize-images-84b9c78fe0ce
    GitHub: https://github.com/josephlee94/intuitive-deep-learning

'''


def main():
    from keras.datasets import cifar10      # Creates exception and crashes program
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print(x_train[0])

    import matplotlib.pyplot as plt

    img = plt.imshow(x_train[0])
    print('The label is:', y_train[0])
    
    img = plt.imshow(x_train[1])
    print('The label is:', y_train[1])

    import keras
    y_train_one_hot = keras.utils.to_categorical(y_train, 10)
    y_test_one_hot = keras.utils.to_categorical(y_test, 10)
    print('The one hot label is:', y_train_one_hot[1])

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255
    x_test = x_test / 255

    x_train[0]

    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32,32,3)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    hist = model.fit(
        x_train,
        y_train_one_hot, 
        batch_size=32,
        epochs=20, 
        validation_split=0.2
    )

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()

    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='lower right')
    plt.show()

    model.evaluate(x_test, y_test_one_hot)[1]

    model.save('my_cifar10_model.h5')

    my_image = plt.imread("cat.jpg")

    from skimage.transform import resize
    my_image_resized = resize(my_image, (32,32,3))

    img = plt.imshow(my_image_resized)

    import numpy as np
    probabilities = model.predict(np.array( [my_image_resized,] ))
    probabilities

    number_to_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    index = np.argsort(probabilities[0,:])
    print("Most likely class:", number_to_class[index[9]], "-- Probability:", probabilities[0,index[9]])
    print("Second most likely class:", number_to_class[index[8]], "-- Probability:", probabilities[0,index[8]])
    print("Third most likely class:", number_to_class[index[7]], "-- Probability:", probabilities[0,index[7]])
    print("Fourth most likely class:", number_to_class[index[6]], "-- Probability:", probabilities[0,index[6]])
    print("Fifth most likely class:", number_to_class[index[5]], "-- Probability:", probabilities[0,index[5]])


main()


# OUTPUT
    # Using TensorFlow backend.
    # Traceback (most recent call last):
    #   File "C:\Users\Scott\AppData\Local\Programs\Python\Python37\lib\site-packages\tensorflow_core\python\pywrap_tensorflow.py", line 58, in <module>
    #     from tensorflow.python.pywrap_tensorflow_internal import *
    #   File "C:\Users\Scott\AppData\Local\Programs\Python\Python37\lib\site-packages\tensorflow_core\python\pywrap_tensorflow_internal.py", line 28, in <module>
    #     _pywrap_tensorflow_internal = swig_import_helper()
    #   File "C:\Users\Scott\AppData\Local\Programs\Python\Python37\lib\site-packages\tensorflow_core\python\pywrap_tensorflow_internal.py", line 24, in swig_import_helper
    #     _mod = imp.load_module('_pywrap_tensorflow_internal', fp, pathname, description)
    #   File "C:\Users\Scott\AppData\Local\Programs\Python\Python37\lib\imp.py", line 242, in load_module
    #     return load_dynamic(name, filename, file)
    #   File "C:\Users\Scott\AppData\Local\Programs\Python\Python37\lib\imp.py", line 342, in load_dynamic
    #     return _load(spec)
    # ImportError: DLL load failed: The specified module could not be found.

    # During handling of the above exception, another exception occurred:

    # Traceback (most recent call last):
    #   File "c:\Users\Scott\.vscode\extensions\ms-python.python-2020.2.64397\pythonFiles\ptvsd_launcher.py", line 48, in <module>
    #     main(ptvsdArgs)
    #   File "c:\Users\Scott\.vscode\extensions\ms-python.python-2020.2.64397\pythonFiles\lib\python\old_ptvsd\ptvsd\__main__.py", line 432, in main
    #     run()
    #   File "c:\Users\Scott\.vscode\extensions\ms-python.python-2020.2.64397\pythonFiles\lib\python\old_ptvsd\ptvsd\__main__.py", line 316, in run_file
    #     runpy.run_path(target, run_name='__main__')
    #   File "C:\Users\Scott\AppData\Local\Programs\Python\Python37\lib\runpy.py", line 263, in run_path
    #     pkg_name=pkg_name, script_name=fname)
    #   File "C:\Users\Scott\AppData\Local\Programs\Python\Python37\lib\runpy.py", line 96, in _run_module_code
    #     mod_name, mod_spec, pkg_name, script_name)
    #   File "C:\Users\Scott\AppData\Local\Programs\Python\Python37\lib\runpy.py", line 85, in _run_code
    #     exec(code, run_globals)
    #   File "c:\Users\Scott\Desktop\Programming\Python_Class\Py_Advanced_Lab_11_Part_2.py", line 118, in <module>
    #     main()
    #   File "c:\Users\Scott\Desktop\Programming\Python_Class\Py_Advanced_Lab_11_Part_2.py", line 17, in main
    #     from keras.datasets import cifar10
    #   File "C:\Users\Scott\AppData\Local\Programs\Python\Python37\lib\site-packages\keras\__init__.py", line 3, in <module>
    #     from . import utils
    #   File "C:\Users\Scott\AppData\Local\Programs\Python\Python37\lib\site-packages\keras\utils\__init__.py", line 6, in <module>
    #     from . import conv_utils
    #   File "C:\Users\Scott\AppData\Local\Programs\Python\Python37\lib\site-packages\keras\utils\conv_utils.py", line 9, in <module>
    #     from .. import backend as K
    #   File "C:\Users\Scott\AppData\Local\Programs\Python\Python37\lib\site-packages\keras\backend\__init__.py", line 1, in <module>
    #     from .load_backend import epsilon
    #   File "C:\Users\Scott\AppData\Local\Programs\Python\Python37\lib\site-packages\keras\backend\load_backend.py", line 90, in <module>
    #     from .tensorflow_backend import *
    #   File "C:\Users\Scott\AppData\Local\Programs\Python\Python37\lib\site-packages\keras\backend\tensorflow_backend.py", line 5, in <module>
    #     import tensorflow as tf
    #   File "C:\Users\Scott\AppData\Local\Programs\Python\Python37\lib\site-packages\tensorflow\__init__.py", line 101, in <module>
    #     from tensorflow_core import *
    #   File "C:\Users\Scott\AppData\Local\Programs\Python\Python37\lib\site-packages\tensorflow_core\__init__.py", line 40, in <module>
    #     from tensorflow.python.tools import module_util as _module_util
    #   File "C:\Users\Scott\AppData\Local\Programs\Python\Python37\lib\site-packages\tensorflow\__init__.py", line 50, in __getattr__
    #     module = self._load()
    #   File "C:\Users\Scott\AppData\Local\Programs\Python\Python37\lib\site-packages\tensorflow\__init__.py", line 44, in _load
    #     module = _importlib.import_module(self.__name__)
    #   File "C:\Users\Scott\AppData\Local\Programs\Python\Python37\lib\importlib\__init__.py", line 127, in import_module
    #     return _bootstrap._gcd_import(name[level:], package, level)
    #   File "C:\Users\Scott\AppData\Local\Programs\Python\Python37\lib\site-packages\tensorflow_core\python\__init__.py", line 49, in <module>
    #     from tensorflow.python import pywrap_tensorflow
    #   File "C:\Users\Scott\AppData\Local\Programs\Python\Python37\lib\site-packages\tensorflow_core\python\pywrap_tensorflow.py", line 74, in <module>
    #     raise ImportError(msg)
    # ImportError: Traceback (most recent call last):
    #   File "C:\Users\Scott\AppData\Local\Programs\Python\Python37\lib\site-packages\tensorflow_core\python\pywrap_tensorflow.py", line 58, in <module>
    #     from tensorflow.python.pywrap_tensorflow_internal import *
    #   File "C:\Users\Scott\AppData\Local\Programs\Python\Python37\lib\site-packages\tensorflow_core\python\pywrap_tensorflow_internal.py", line 28, in <module>
    #     _pywrap_tensorflow_internal = swig_import_helper()
    #   File "C:\Users\Scott\AppData\Local\Programs\Python\Python37\lib\site-packages\tensorflow_core\python\pywrap_tensorflow_internal.py", line 24, in swig_import_helper
    #     _mod = imp.load_module('_pywrap_tensorflow_internal', fp, pathname, description)
    #   File "C:\Users\Scott\AppData\Local\Programs\Python\Python37\lib\imp.py", line 242, in load_module
    #     return load_dynamic(name, filename, file)
    #   File "C:\Users\Scott\AppData\Local\Programs\Python\Python37\lib\imp.py", line 342, in load_dynamic
    #     return _load(spec)
    # ImportError: DLL load failed: The specified module could not be found.


    # Failed to load the native TensorFlow runtime.

    # See https://www.tensorflow.org/install/errors

    # for some common reasons and solutions.  Include the entire stack trace
    # above this error message when asking for help.
