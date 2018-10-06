from keras import backend as K

__all__ = ['categorical', 'sequence']

def categorical(y_true, y_pred):
    return -K.categorical_crossentropy(y_true, y_pred)

def sequence(y_true, y_pred):
    y_shape = K.shape(y_true)
    y_true = K.reshape(y_true, (-1, y_shape[-1]))
    y_pred = K.reshape(y_pred, (-1, y_shape[-1]))
    LL = -K.categorical_crossentropy(y_true, y_pred)
    LL = K.sum(K.reshape(LL, (-1, y_shape[1])), axis=1)
    return LL
