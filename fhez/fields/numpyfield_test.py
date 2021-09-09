# @Author: George Onoufriou <archer>
# @Date:   2021-08-22T16:32:08+01:00
# @Last modified by:   archer
# @Last modified time: 2021-09-09T15:42:30+01:00

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

    def test_serialDeserialFloat(self):
        """Check serialisation and deserialisation working properly."""
        # set up necessary vars
        data = self.data
        sample = {"data": data}
        schema = mar.Schema.from_dict({"data": NumpyField()})

        # now test serialisation
        out = schema().dump(sample)

        # now test deserialisation
        deserial = schema().load(out)

        # print("DESERIAL OUTPUT TYPE: {}".format(type(out)))
        self.assertIsInstance(out, dict)

        # check equality
        np.testing.assert_array_almost_equal(
            sample["data"],
            deserial["data"],
            decimal=1,
            verbose=True)

        # check type is as original/ expected
        # self.assertTrue(np.issubdtype(sample["data"].dtype, np.floating))
        self.assertEqual(sample["data"].dtype, sample["data"].dtype)

    def test_serialDeserialsFloat(self):
        """Check serialisation and deserialisation working properly."""
        # set up necessary vars
        data = self.data
        sample = {"data": data}
        schema = mar.Schema.from_dict({"data": NumpyField()})

        # now test serialisation
        out = schema().dumps(sample)

        # print("DESERIALS OUTPUT TYPE: {}".format(type(out)))
        self.assertIsInstance(out, str)

        # now test deserialisation
        deserial = schema().loads(out)

        # check equality
        np.testing.assert_array_almost_equal(
            sample["data"],
            deserial["data"],
            decimal=1,
            verbose=True)

        # check type is as original/ expected
        # self.assertTrue(np.issubdtype(sample["data"].dtype, np.floating))
        self.assertEqual(sample["data"].dtype, sample["data"].dtype)

    def test_serialDeserialInt(self):
        """Check serialisation and deserialisation working properly."""
        # set up necessary vars
        data = self.data.astype(int)
        # all of the data in the sample will inevitably be 0 since trunication
        sample = {"data": data}
        schema = mar.Schema.from_dict({"data": NumpyField()})

        # now test serialisation
        out = schema().dump(sample)

        # print("DESERIAL OUTPUT TYPE: {}".format(type(out)))
        self.assertIsInstance(out, dict)

        # now test deserialisation
        deserial = schema().load(out)

        # check equality
        np.testing.assert_array_almost_equal(
            sample["data"],
            deserial["data"],
            decimal=1,
            verbose=True)

        # check type is as original/ expected
        # self.assertTrue(np.issubdtype(sample["data"].dtype, np.integer))
        self.assertEqual(sample["data"].dtype, sample["data"].dtype)
