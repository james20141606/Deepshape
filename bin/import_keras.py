# import keras modules with modified TensorFlow configuration
def _get_session():
    """Modified the original get_session() function to change the ConfigProto variable
    """
    global _SESSION
    if tf.get_default_session() is not None:
        session = tf.get_default_session()
    else:
        if _SESSION is None:
            if not os.environ.get('OMP_NUM_THREADS'):
                config = tf.ConfigProto(allow_soft_placement=True)
            else:
                nb_thread = int(os.environ.get('OMP_NUM_THREADS'))
                config = tf.ConfigProto(intra_op_parallelism_threads=nb_thread,
                                        allow_soft_placement=True)
            config.gpu_options.per_process_gpu_memory_fraction = 0.8
            config.gpu_options.allow_growth = True
            _SESSION = tf.Session(config=config)
        session = _SESSION
    if not _MANUAL_VAR_INIT:
        _initialize_variables()
    return session

def import_keras():
    """Import the heavy modules after command line parsing to accelerate startup process
    """
    import keras
    from keras.models import Sequential, Model
    from keras.layers import Input, Activation, Dense
    from keras.layers.core import RepeatVector, Reshape, Flatten, Dropout, Lambda
    from keras.layers.convolutional import Conv1D, Conv2D, UpSampling1D, UpSampling2D
    from keras.layers.pooling import MaxPooling1D, MaxPooling2D, AveragePooling1D, AveragePooling2D
    from keras.regularizers import l2, l1, l1_l2
    from keras.layers.merge import Add, Concatenate
    from keras.layers.normalization import BatchNormalization
    from keras.optimizers import Adam, RMSprop, SGD
    from keras import backend as K
    import keras.backend.tensorflow_backend
    # control GPU memory usage for TensorFlow backend
    if K.backend() == 'tensorflow':
        # replace the original get_session() function
        keras.backend.tensorflow_backend.get_session.func_code = _get_session.func_code
        import tensorflow as tf

    import numpy as np
    globals().update(locals())

def patch_keras():
    def binary_crossentropy_with_nan(y_true, y_pred):
        not_nan = tf.logical_not(tf.is_nan(y_true))
        y_true = tf.boolean_mask(y_true, not_nan)
        y_pred = tf.boolean_mask(y_pred, not_nan)
        return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)
    keras.losses.binary_crossentropy_with_nan = binary_crossentropy_with_nan

    def binary_accuracy_with_nan(y_true, y_pred):
        not_nan = tf.logical_not(tf.is_nan(y_true))
        y_true = tf.boolean_mask(y_true, not_nan)
        y_pred = tf.boolean_mask(y_pred, not_nan)
        return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)
    keras.metrics.binary_accuracy_with_nan = binary_accuracy_with_nan

    def mse_with_nan(y_true, y_pred):
        not_nan = tf.logical_not(tf.is_nan(y_true))
        y_true = tf.boolean_mask(y_true, not_nan)
        y_pred = tf.boolean_mask(y_pred, not_nan)
        return K.mean(K.square(y_pred - y_true), axis=-1)
    keras.metrics.mse_with_nan = mse_with_nan
    keras.losses.mse_with_nan = mse_with_nan
