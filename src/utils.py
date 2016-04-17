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
    return {x:getattr(namespace, x) for x in attrs}

def sub_dict(dictionary, keys):
    """
    Get part of a dict
    """
    return {x:dictionary[x] for x in keys}

class SparseRows():
    """
    Sparse rows
    """
    __slots__ = ('indices', 'array', 'next')
    
    def __init__(self, shape):
        self.indices = np.empty(shape[0], dtype='int')
        self.array = np.empty(shape)
        self.next = 0
    
    def __setitem__(self, key, value):
        self.indices[self.next] = key
        self.array[self.next] = value
        self.next += 1 
    
    def add_to(self, other):
        assert self.next == self.indices.shape[0]
        other[self.indices] += self.array

def sparse_like(matrix, num_rows):
    """
    Create a SparseRows object based on a matrix, with a specified number of rows
    """
    shape = list(matrix.shape)
    shape[0] = num_rows
    return SparseRows(shape)