import numpy as np, signal, traceback
from multiprocessing import Array
from bisect import bisect_left

# Shared numpy arrays

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

# Row-sparse matrices

class SparseRows():
    """
    Sparse rows
    """
    __slots__ = ('indices', 'array', 'next')
    
    def __init__(self, shape):
        """
        Initialise a sparse numpy array with a given shape
        :param shape: tuple of integers, where the first index is the number of *non-zero* rows
        """
        self.indices = np.empty(shape[0], dtype='int')
        self.array = np.empty(shape)
        self.next = 0
    
    def __setitem__(self, key, value):
        """
        Set the value of the next non-zero row
        :param key: index of the row
        :param value: value of the row
        """
        self.indices[self.next] = key
        self.array[self.next] = value
        self.next += 1 
    
    def add_to(self, other):
        """
        Add this array to a non-sparse array 
        :param other: numpy array
        """
        if self.next != self.indices.shape[0]:
            raise Exception('SparseRows object has not been filled')
        other[self.indices] += self.array

def sparse_like(matrix, num_rows):
    """
    Create a SparseRows object based on a matrix, with a specified number of rows
    """
    shape = list(matrix.shape)
    shape[0] = num_rows
    return SparseRows(shape)

# Getting subparts

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

# Miscellaneous

def is_verb(string):
    """
    Check if a predstring is for a verb or a noun
    """
    return string.split('_')[-2] == 'v'

def product(iterable):
    """
    Calculate the product of the elements of an iterable
    """
    res = 1
    for x in iterable:
        res *= x
    return res

def index(sorted_list, x):
    """
    Locate the first value in sorted_list exactly equal to x
    """
    i = bisect_left(sorted_list, x)
    if i != len(sorted_list) and sorted_list[i] == x:
        return i
    raise ValueError(x)

def cosine(u,v):
    """
    Calculate the cosine similarity between two vectors
    :param u: numpy array
    :param v: numpy array
    :return: similarity
    """
    return np.dot(u,v) / np.sqrt(np.dot(u,u) * np.dot(v,v))

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

def alias_sample(U, K, n=None):
    """
    Sample from a categorical distribution, using the alias method
    :param U: probability table
    :param K: alias table
    :param n: number of samples to draw (int or tuple of ints)
    :return: array of samples
    """
    if n:
        # Choose random indices
        i = np.random.randint(U.size, size=n)
        # Choose whether to return indices or aliases
        switch = (np.random.random(n) > U[i])
        return switch * K[i] + np.invert(switch) * i
    else:
        return alias_sample_one(U, K)

# Converting pydmrs data to form required by SemFuncModel

def reform_links(self, node, ents):
    """
    Get the links from a PointerNode,
    and convert them to the form required by SemFuncModel
    :param node: a PointerNode
    :param ents: a matrix of entity vectors (indexed by nodeid)
    """
    out_labs = []
    out_vecs = []
    in_labs = []
    in_vecs = []
    for l in node.get_out(itr=True):
        out_labs.append(l.rargname)
        out_vecs.append(ents[l.end])
    for l in node.get_in(itr=True):
        in_labs.append(l.rargname)
        in_vecs.append(ents[l.start])
    return out_labs, out_vecs, in_labs, in_vecs

def reform_out_links(self, node, ents):
    """
    Get the outgoing links from a PointerNode,
    and convert them to the form required by SemFuncModel
    :param node: a PointerNode
    :param ents: a matrix of entity vectors (indexed by nodeid)
    """
    out_labs = []
    out_vecs = []
    for l in node.get_out(itr=True):
        out_labs.append(l.rargname)
        out_vecs.append(ents[l.end])
    return out_labs, out_vecs

# Running code with a timeout

class UserTimeoutError(Exception):
    """
    Errors thrown by Timeout context manager
    """

class Timeout:
    """
    Context manager that allows running code with a timeout
    """
    def __init__(self, seconds=1, error_message=None):
        """
        Initialise settings
        :param seconds: number of seconds to wait
        :param error_message: message to include in UserTimeoutError
        """
        self.seconds = seconds
        if error_message is not None:
            self.error_message = error_message
        else:
            # Default error message
            self.error_message = 'Timeout after {} second'.format(seconds)
            if seconds is not 1:  # Pluralise if not 1 second 
                self.error_message += 's'

    def handle_timeout(self, signum, frame):
        """
        Raise an error - function to be called by signal module
        :param signum: signal that was sent
        :param frame: stack frame
        """
        # Get the raw traceback from the current stack frame
        stack = traceback.extract_stack(frame)
        # Raise an error with the message and the traceback
        raise UserTimeoutError(self.error_message, stack)

    def __enter__(self):
        """
        Enter the context
        """
        # Listen to SIGALRM with self.handle_timeout
        signal.signal(signal.SIGALRM, self.handle_timeout)
        # Schedule a signal to be sent after specified number of seconds
        signal.alarm(self.seconds)

    def __exit__(self, exception_type, exception_value, traceback):
        """
        Exit the context
        """
        # Cancel the scheduled signal
        signal.alarm(0)