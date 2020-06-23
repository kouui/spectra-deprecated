Numpy Cheat Sheet
==========================

Linear algebra
------------------

* `numpy.kron(a,b)`_

    Kronecker product of two arrays.

    Examples:

    >>> np.kron(np.eye(2), np.ones((2,2)))
    array([[ 1.,  1.,  0.,  0.],
           [ 1.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  1.],
           [ 0.,  0.,  1.,  1.]])

In linear system :math:`Ax = b`, if A is a full matrix, system of equations can always be solved by
using `numpy.linalg.solve()`_, or `numpy.linalg.inv()`_. However, if A is arbitrary matrix,
i.e. under-, well-, or over- determined, we'd better solve the system of equations by least-squares method,
i.e. the following two functions

* `numpy.linalg.lstsq(a, b, rcond='warn')`_

    Return the least-squares solution to a linear matrix equation.

    Examples:

    >>> x = np.array([0, 1, 2, 3])
    >>> y = np.array([-1, 0.2, 0.9, 2.1])
    >>> A = np.vstack([x, np.ones(len(x))]).T
    >>> A
    array([[ 0.,  1.],
           [ 1.,  1.],
           [ 2.,  1.],
           [ 3.,  1.]])
    >>> m, c = np.linalg.lstsq(A, y)[0]
    >>> print(m, c)
    1.0 -0.95

* `numpy.linalg.pinv(a, rcond=1e-15)`_

    Calculate the generalized inverse of a matrix using its
    singular-value decomposition (SVD) and including all large singular values.

    Examples:

    >>> a = np.random.randn(9, 6)
    >>> B = np.linalg.pinv(a)
    >>> np.allclose(a, np.dot(a, np.dot(B, a)))
    True
    >>> np.allclose(B, np.dot(B, np.dot(a, B)))
    True

To see whether a linear system can be easly solve numerically, we always look
into the condition number of matrix A,

* `numpy.linalg.cond(x, p=None)[source]`_

    This function is capable of returning the condition number
    using one of seven different norms, depending on the value of p.

    Example:

    >>> from numpy import linalg as LA
    >>> a = np.array([[1, 0, -1], [0, 1, 0], [1, 0, 1]])
    >>> a
    array([[ 1,  0, -1],
           [ 0,  1,  0],
           [ 1,  0,  1]])
    >>> LA.cond(a)
    1.4142135623730951



Searching and Sorting
----------------------

The most used searching function is `numpy.where()`_. Besides, There are many other useful functions.

* `numpy.unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None)`_

    Find the unique elements of an array.

    Example:

    >>> a = np.array([[1, 1], [2, 3]])
    >>> np.unique(a)
    array([1, 2, 3])
    >>> a = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]])
    >>> np.unique(a, axis=0)
    array([[1, 0, 0], [2, 3, 4]])

while `numpy.sort()`_ returns a sorted array, if you need the order of
how computer sort your array, you should use

* `numpy.argsort(a, axis=-1, kind='quicksort', order=None)`_

    Returns the indices that would sort an array.

    Example:

    >>> x = np.array([3, 1, 2])
    >>> np.argsort(x)
    array([1, 2, 0])

Iterating
-----------

Rarely used but seems interesting. For details, see the `introductory guide to array iteration`_

* `numpy.ndenumerate(arr)`_

    class, returning an iterator yielding pairs of array coordinates and values.

    Example:

    >>> a = np.array([[1, 2], [3, 4]])
    >>> for index, x in np.ndenumerate(a):
    ...     print(index, x)
    (0, 0) 1
    (0, 1) 2
    (1, 0) 3
    (1, 1) 4

* `numpy.ndindex(*shape)`_

    iterates over the N-dimensional index of the array.
    At each iteration a tuple of indices is returned,
    the last dimension is iterated over first.

    Example:

    >>> for index in np.ndindex(3, 2, 1):
    ...     print(index)
    (0, 0, 0)
    (0, 1, 0)
    (1, 0, 0)
    (1, 1, 0)
    (2, 0, 0)
    (2, 1, 0)

`Standard Universal Function`_
------------------------------
There are currently more than 60 universal functions defined in ``numpy`` on one or more types,
covering a wide variety of operations. Some of these ufuncs are called automatically on arrays
when the relevant infix notation is used (e.g., ``add(a, b)`` is called internally
when ``a + b`` is written and a or b is an ``ndarray``).
Nevertheless, you may still want to use the ufunc call in order to use the optional
output argument(s) to place the output(s) in an object (or objects) of your choice.

**Tip:**

The optional output arguments can be used to help you save memory for large calculations.
If your arrays are large, complicated expressions can take longer than absolutely necessary
due to the creation and (later) destruction of temporary calculation spaces.

For example, the expression ``G = a * b + c`` is equivalent to ``t1 = A * B; G = T1 + C; del t1``.
It will be more quickly executed as ``G = A * B; add(G, C, G)`` which is the same as ``G = A * B; G += C``.





.. _numpy.kron(a,b): https://docs.scipy.org/doc/numpy/reference/generated/numpy.kron.html#numpy.kron
.. _numpy.linalg.solve(): http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.solve.html#numpy.linalg.solve
.. _numpy.linalg.inv(): http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.inv.html#numpy.linalg.inv
.. _numpy.linalg.lstsq(a, b, rcond='warn'): https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html#numpy.linalg.lstsq
.. _numpy.linalg.pinv(a, rcond=1e-15): https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.pinv.html#numpy.linalg.pinv
.. _numpy.linalg.cond(x, p=None)[source]: https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.cond.html#numpy.linalg.cond
.. _numpy.where(): https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html#numpy.where
.. _numpy.unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None): https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html#numpy.unique
.. _numpy.sort(): https://docs.scipy.org/doc/numpy/reference/generated/numpy.sort.html
.. _numpy.argsort(a, axis=-1, kind='quicksort', order=None): https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
.. _numpy.ndenumerate(arr): https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndenumerate.html#numpy.ndenumerate
.. _numpy.ndindex(*shape): https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndindex.html#numpy.ndindex
.. _introductory guide to array iteration: https://docs.scipy.org/doc/numpy/reference/arrays.nditer.html#arrays-nditer
.. _Standard Universal Function: https://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs
