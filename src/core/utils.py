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

def shared_zeros(shape, ctype='d', dtype='float64'):
    """
    Initialise a shared numpy array, filled with zeros
    """
    # Create a flat shared array
    size = int(np.prod(shape))
    shared = Array(ctype, size)
    # Create a new numpy array from the shared array
    flat_array = np.frombuffer(shared._obj, dtype)
    # Reshape the new array
    return flat_array.reshape(shape)

def shared_zeros_like(array):
    """
    Initialise a shared numpy array, filled with zeros, like another array
    """
    # Get the data types for the array
    dtype = array.dtype
    ctype = np.ctypeslib.as_ctypes(array)
    while not isinstance(ctype, str): ctype = ctype._type_
    # Initialise the array
    return shared_zeros(array.shape, ctype, dtype)

def is_verb(string):
    """
    Check if a predstring is for a verb or a noun
    """
    return string.split('_')[-2] == 'v'

def sub_namespace(namespace, attrs, strict=False):
    """
    Get part of a namespace, as a dict
    """
    if strict:
        return {x:getattr(namespace, x) for x in attrs}
    else:
        return {x:getattr(namespace, x) for x in attrs if hasattr(namespace, x)}

def sub_dict(dictionary, keys, strict=False):
    """
    Get part of a dict
    """
    if strict:
        return {x:dictionary[x] for x in keys}
    else:
        return {x:dictionary[x] for x in keys if x in dictionary}

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

def product(iterable):
    """
    Calculate the product of the elements of an iterable
    """
    res = 1
    for x in iterable:
        res *= x
    return res

# The alias method samples from a categorical distribution in O(1) time
# https://en.wikipedia.org/wiki/Alias_method

def init_alias(prob):
    """
    Initialise arrays for sampling with the alias method
    :param prob: probability array
    :return: probability table, alias table
    """
    N = prob.size
    # Initialise tables
    U = prob.astype('float64') / prob.sum() * N
    K = np.arange(N)
    # Initialise lists with weight above and below 1
    below = [i for i,x in enumerate(U) if x<1]
    above = [i for i,x in enumerate(U) if x>=1]
    # Fill tables
    # In each iteration, we remove one index from the pair of lists
    while above and below:
        # Take a pair of indices, one above and one below
        i = below.pop()
        j = above.pop()
        # Fill in the tables
        K[i] = j
        # Calculate the remaining weight of j, and put it back in the correct list
        U[j] -= (1 - U[i])
        if U[j] < 1:
            below.append(j)
        else:
            above.append(j)
    # Note the final index will have U=1, up to rounding error
    return U, K

def alias_sample_one(U, K):
    """
    Sample from a categorical distribution, using the alias method
    :param U: probability table
    :param K: alias table
    :return: sample
    """
    # Choose a random index
    i = np.random.randint(U.size)
    # Return the index, or the alias
    if np.random.rand() > U[i]:
        return K[i]
    else:
        return i

def alias_sample(U, K, n):
    """
    Sample from a categorical distribution, using the alias method
    :param U: probability table
    :param K: alias table
    :param n: number of samples to draw
    :return: array of samples
    """
    # Choose random indices
    i = np.random.randint(U.size, size=n)
    # Choose whether to return indices or aliases
    switch = (np.random.rand(n) > U[i])
    return switch * K[i] + np.invert(switch) * i
