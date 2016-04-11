from multiprocessing import Array
import numpy as np

def make_shared(array):
    """
    Convert a numpy array to a multiprocessing array with numpy access
    """
    # Get the C type for the array
    ctype = np.ctypeslib.as_ctypes(array)
    while not isinstance(ctype, str): ctype = ctype._type_
    # Create a shared array
    shared = Array(ctype, array.flatten())
    # Create a new numpy array from the shared array
    flat_array = np.frombuffer(shared._obj, #._wrapper.create_memoryview(),
                               dtype = array.dtype)
    # Reshape the new array
    return flat_array.reshape(array.shape)

def is_verb(string):
    """
    Check if a predstring is for a verb or a noun
    """
    return string.split('_')[-2] == 'v'

def sub_namespace(namespace, attrs):
    """
    Get part of a namespace, as a dict
    """
    subspace = {}
    for x in attrs:
        subspace[x] = getattr(namespace, x)
    return subspace