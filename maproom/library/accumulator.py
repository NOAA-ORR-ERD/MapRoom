#!/usr/bin/env python

"""
accumulator class

Designed to be used as an expandable numpy array, to accumulate values, rather
than a python list.

Note that slices return copies, rather than views, unlike regular numpy arrays.
This is so that the buffer can be re-allocated without messing up any views.

IMPORTAT Only works for 1-d arrays at the moment.
"""
import numpy as np


class accumulator(object):
    # A few parameters
    DEFAULT_BUFFER_SIZE = 128
    BUFFER_EXTEND_SIZE = 1.25  # array.array uses 1+1/16 -- that seems small to me.

    def __init__(self, object=None, dtype=np.float, block_shape=()):
        """
        proper docs here

        note: a scalar accumulator doesn't really make sense, so you get a length - 1 array instead.

        block_shape specifies the other dimensions of the array, so that it will be of shape:
          (n,) + block_shape
        block_shape is ignored if object is provided, and the shape of the array
        is determined from the shape of the provided object.

        If neither object nor block_shape is provided, and empty 1-d array is created
        """
        if object is None:
            buffer = np.empty((0,) + block_shape, dtype=dtype)
        else:
            buffer = np.array(object, dtype=dtype, copy=True)
            if buffer.shape == ():  # to make sure we don't have a scalar
                buffer.shape = (1,)
        self._length = buffer.shape[0]
        self._block_shape = buffer.shape[1:]
        # add the padding to the buffer
        shape = (max(self.DEFAULT_BUFFER_SIZE, buffer.shape[0] * self.BUFFER_EXTEND_SIZE), ) + buffer.shape[1:]
        buffer.resize(shape)
        self.__buffer = buffer

    # fixme:
    # using @property seems to give a getter, but setting then overrides it
    # which seems terribly prone to error.
    @property
    def dtype(self):
        return self.__buffer.dtype

    @property
    def bufferlength(self):
        """
        the size of the internal buffer
        """
        return self.__buffer.shape[0]

    @property
    def shape(self):
        """
        To be compatible with ndarray.shape
        (only the getter!)
        """
        return (self._length,) + self._block_shape

    def __len__(self):
        return self._length

    def __array__(self, dtype=None):
        """
        a.__array__(|dtype) -> copy of array.

        Always returns a copy array, so that buffer doesn't have any references to it.
        """
        return np.array(self.__buffer[:self._length], dtype=dtype, copy=True)

    def append(self, item):
        """
        add a new item to the end of the array.

        It should be one less dimension than the array: i.e. a.shape[1:]
        if the itme is a smaller shape, it needs to be broadcastable to that shape.
        """
        try:
            self.__buffer[self._length] = item
            self._length += 1
        except IndexError:  # the buffer is not big enough or wrong shape entries
            # fixme: test for wrong shape?
            self.resize(self._length * self.BUFFER_EXTEND_SIZE,)
            self.append(item)

    def extend(self, items):
        """
        add a sequence of new items to the end of the array
        """
        try:
            self.__buffer[self._length:self._length + len(items)] = items
            self._length += len(items)
        except ValueError:  # the buffer is not big enough, or wrong shape
            items = np.asarray(items, dtype=self.dtype)
            if items.shape[1:] != self._block_shape:
                raise
            self.resize((self._length + len(items)) * self.BUFFER_EXTEND_SIZE)
            self.extend(items)

    def resize(self, newsize):
        """
        resize the internal buffer

        it takes a scalar for the length of the the first axis appropriately.

        You might want to do this to speed things up if you know you want it
        to be a lot bigger eventually
        """
        newsize = int(newsize)
        if newsize < self._length:
            raise ValueError("accumulator buffer cannot be made smaller than the data")
        shape = (newsize,) + self._block_shape
        self.__buffer.resize(shape)

    def fitbuffer(self):
        """
        re-sizes the buffer so that it fits the data, rather than having extra space

        """
        self.__buffer.resize((self._length,) + self._block_shape)

    def __getitem__(self, index):
        # fixme -- this needs to be expanded to n-d!
        if index > self._length - 1:
            raise IndexError("index out of bounds")
        elif index < 0:
            index = self._length - 1
        return self.__buffer[index]

    # apparently __getslice__ is depricated!
    def __getslice__(self, i, j):
        """
        a.__getslice__(i, j) <==> a[i:j]

        Use of negative indices is not supported.

        This returns a COPY, not a view, unlike numpy arrays
        This is required as the data buffer needs to be able to change.
        """
        # fixme -- this needs to be updated: it should be in __getitem__
        #         and support 2d
        j = min(j, self._length)
        return self.__buffer[i:j].copy()

    def __delitem__(self):
        raise NotImplementedError

    def __eq__(self, other):
        return self.__buffer[:self._length] == other
    # fixme: other comparison method here

    def __str__(self):
        return self.__buffer[:self.shape[0]].__str__()

    def __repr__(self):
        return "accumulator%s" % self.__buffer[:self.shape[0]].__repr__()[5:]


def flatten(list_of_lists):
    return [j for i in list_of_lists for j in i]


if __name__ == "__main__":
    polygon_points = accumulator(block_shape=(2,), dtype=np.float64)
    p = [(0, 0)] * 190
    polygon_points.extend(p)  # works
    p = [(0, 0, 0)] * 190
    polygon_points.extend(p)  # fails with ValueError, even though it's supposed to catch ValueError!
