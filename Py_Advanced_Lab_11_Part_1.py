''' INTRO

Name: Scott Qualkenbush

Course: CIS-2532-NET01

Date: 5-10-2020

Program Summary: 

    Deep Learning exploration, Part 1: Build your first Neural Network to predict house prices with Keras

    Code credit goes to Joseph Lee Wei En

    Source: https://medium.com/intuitive-deep-learning/build-your-first-neural-network-to-predict-house-prices-with-keras-eb5db60232c
    GitHub: https://github.com/josephlee94/intuitive-deep-learning

'''


def main():
    import pandas as pd

    df = pd.read_csv('housepricedata.csv')
    
    dataset = df.values
    
    X = dataset[:,0:10]
    Y = dataset[:,10]

    from sklearn import preprocessing
    
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scale = min_max_scaler.fit_transform(X)
    
    from sklearn.model_selection import train_test_split

    X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

    from keras.models import Sequential     # Creates exception and crashes program
    from keras.layers import Dense          # Creates exception and crashes program

    model = Sequential([
        Dense(32, activation='relu', input_shape=(10,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer='sgd',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    hist = model.fit(
        X_train,
        Y_train,
        batch_size=32,
        epochs=100,
        validation_data=(X_val, Y_val)
    )

    model.evaluate(X_test, Y_test)[1]

    import matplotlib.pyplot as plt

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

    model_2 = Sequential([
        Dense(1000, activation='relu', input_shape=(10,)),
        Dense(1000, activation='relu'),
        Dense(1000, activation='relu'),
        Dense(1000, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])

    model_2.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']    
    )
    
    hist_2 = model_2.fit(
        X_train,
        Y_train,
        batch_size=32,
        epochs=100,
        validation_data=(X_val, Y_val)
    )

    plt.plot(hist_2.history['loss'])
    plt.plot(hist_2.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()

    plt.plot(hist_2.history['acc'])
    plt.plot(hist_2.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='lower right')
    plt.show()

    from keras.layers import Dropout
    from keras import regularizers

    model_3 = Sequential([
        Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(10,)),
        Dropout(0.3),
        Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dropout(0.3),
        Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dropout(0.3),
        Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dropout(0.3),
        Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)),
    ])

    model_3.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    hist_3 = model_3.fit(
        X_train,
        Y_train,
        batch_size=32,
        epochs=100,
        validation_data=(X_val, Y_val)
    )

    plt.plot(hist_3.history['loss'])
    plt.plot(hist_3.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.ylim(top=1.2, bottom=0)
    plt.show()

    plt.plot(hist_3.history['acc'])
    plt.plot(hist_3.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='lower right')
    plt.show()


main()


# OUTPUT
    # Using TensorFlow backend.
    # Traceback (most recent call last):
    # File "C:\Users\Scott\anaconda3\lib\site-packages\tensorflow\python\pywrap_tensorflow.py", line 58, in <module>
    #     from tensorflow.python.pywrap_tensorflow_internal import *
    # File "C:\Users\Scott\anaconda3\lib\site-packages\tensorflow\python\pywrap_tensorflow_internal.py", line 28, in <module>
    #     _pywrap_tensorflow_internal = swig_import_helper()
    # File "C:\Users\Scott\anaconda3\lib\site-packages\tensorflow\python\pywrap_tensorflow_internal.py", line 24, in swig_import_helper
    #     _mod = imp.load_module('_pywrap_tensorflow_internal', fp, pathname, description)
    # File "C:\Users\Scott\anaconda3\lib\imp.py", line 242, in load_module
    #     return load_dynamic(name, filename, file)
    # File "C:\Users\Scott\anaconda3\lib\imp.py", line 342, in load_dynamic
    #     return _load(spec)
    # ImportError: DLL load failed: The specified module could not be found.

    # During handling of the above exception, another exception occurred:

    # Traceback (most recent call last):
    # File "c:\Users\Scott\.vscode\extensions\ms-python.python-2020.2.64397\pythonFiles\ptvsd_launcher.py", line 48, in <module>
    #     main(ptvsdArgs)
    # File "c:\Users\Scott\.vscode\extensions\ms-python.python-2020.2.64397\pythonFiles\lib\python\old_ptvsd\ptvsd\__main__.py", line 432, in main
    #     run()
    # File "c:\Users\Scott\.vscode\extensions\ms-python.python-2020.2.64397\pythonFiles\lib\python\old_ptvsd\ptvsd\__main__.py", line 316, in run_file
    #     runpy.run_path(target, run_name='__main__')
    # File "C:\Users\Scott\anaconda3\lib\runpy.py", line 263, in run_path
    #     pkg_name=pkg_name, script_name=fname)
    # File "C:\Users\Scott\anaconda3\lib\runpy.py", line 96, in _run_module_code
    #     mod_name, mod_spec, pkg_name, script_name)
    # File "C:\Users\Scott\anaconda3\lib\runpy.py", line 85, in _run_code
    #     exec(code, run_globals)
    # File "c:\Users\Scott\Desktop\Programming\Python_Class\Py_Advanced_Lab_11_Part_1.py", line 164, in <module>
    #     main()
    # File "c:\Users\Scott\Desktop\Programming\Python_Class\Py_Advanced_Lab_11_Part_1.py", line 36, in main
    #     from keras.models import Sequential     # Creates exception and crashes program
    # File "C:\Users\Scott\anaconda3\lib\site-packages\keras\__init__.py", line 3, in <module>
    #     from . import utils
    # File "C:\Users\Scott\anaconda3\lib\site-packages\keras\utils\__init__.py", line 6, in <module>
    #     from . import conv_utils
    # File "C:\Users\Scott\anaconda3\lib\site-packages\keras\utils\conv_utils.py", line 9, in <module>
    #     from .. import backend as K
    # File "C:\Users\Scott\anaconda3\lib\site-packages\keras\backend\__init__.py", line 1, in <module>
    #     from .load_backend import epsilon
    # File "C:\Users\Scott\anaconda3\lib\site-packages\keras\backend\load_backend.py", line 90, in <module>
    #     from .tensorflow_backend import *
    # File "C:\Users\Scott\anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py", line 5, in <module>
    #     import tensorflow as tf
    # File "C:\Users\Scott\anaconda3\lib\site-packages\tensorflow\__init__.py", line 41, in <module>
    #     from tensorflow.python.tools import module_util as _module_util
    # File "C:\Users\Scott\anaconda3\lib\site-packages\tensorflow\python\__init__.py", line 50, in <module>
    #     from tensorflow.python import pywrap_tensorflow
    # File "C:\Users\Scott\anaconda3\lib\site-packages\tensorflow\python\pywrap_tensorflow.py", line 69, in <module>
    #     raise ImportError(msg)
    # ImportError: Traceback (most recent call last):
    # File "C:\Users\Scott\anaconda3\lib\site-packages\tensorflow\python\pywrap_tensorflow.py", line 58, in <module>
    #     from tensorflow.python.pywrap_tensorflow_internal import *
    # File "C:\Users\Scott\anaconda3\lib\site-packages\tensorflow\python\pywrap_tensorflow_internal.py", line 28, in <module>
    #     _pywrap_tensorflow_internal = swig_import_helper()
    # File "C:\Users\Scott\anaconda3\lib\site-packages\tensorflow\python\pywrap_tensorflow_internal.py", line 24, in swig_import_helper
    #     _mod = imp.load_module('_pywrap_tensorflow_internal', fp, pathname, description)
    # File "C:\Users\Scott\anaconda3\lib\imp.py", line 242, in load_module
    #     return load_dynamic(name, filename, file)
    # File "C:\Users\Scott\anaconda3\lib\imp.py", line 342, in load_dynamic
    #     return _load(spec)
    # ImportError: DLL load failed: The specified module could not be found.


    # Failed to load the native TensorFlow runtime.

    # See https://www.tensorflow.org/install/errors

    # for some common reasons and solutions.  Include the entire stack trace
    # above this error message when asking for help.


