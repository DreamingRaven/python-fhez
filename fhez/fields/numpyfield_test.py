# @Author: George Onoufriou <archer>
# @Date:   2021-08-22T16:32:08+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-22T17:53:44+01:00

import time
import unittest
import numpy as np
import marshmallow as mar
from fhez.fields.numpyfield import NumpyField


class NumpyFieldTest(unittest.TestCase):
    """Test Numpy field."""

    @property
    def data_shape(self):
        """Define desired data shape."""
        return (3, 32, 32, 3)

    @property
    def data(self):
        """Get some generated data."""
        array = np.random.rand(*self.data_shape)
        return array

    def setUp(self):
        """Start timer and init variables."""
        self.weights = (1,)  # if tuple allows cnn to initialise itself

        self.startTime = time.time()

    def tearDown(self):
        """Calculate and print time delta."""
        t = time.time() - self.startTime
        print('%s: %.3f' % (self.id(), t))

    def testSerialDeserial(self):
        """Check serialisation and deserialisation working properly."""
        # set up necessary vars
        data = self.data
        sample = {"data": data}
        schema = mar.Schema.from_dict({"data": NumpyField()})

        # now test serialisation
        out = schema().dump(sample)

        # now test deserialisation
        deserial = schema().load(out)
