"""Marshmallow numpy field."""
# @Author: George Onoufriou <archer>
# @Date:   2021-08-20T16:38:06+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-22T17:51:57+01:00

import numpy as np
import marshmallow as mar


class NumpyField(mar.fields.Field):
    """Marshmallow field to serialise and deserialise numpy data."""

    schema = {
        "dtype": mar.fields.Str(),
        "data": mar.fields.List(mar.fields.Float())
    }

    def _serialize(self, value, attr, obj, **kwargs):
        """Use marshmallow to serialise numpy as list of numbers."""
        if value is None:
            return None

        # define and set the to be stored data
        store = {
            "dtype": value.dtype.name,
            "data": value.tolist()
        }
        # schema = mar.Schema.from_dict(NumpyField.schema)
        # now used marshmallow existing handler for list of our numpy
        # serialised = schema().dump(store)
        return store

    def _deserialize(self, value, attr, data, **kwargs):
        """Use marshmallow to deserialise list of numbers back to numpy."""
        if value is None:
            return None

        # schema = mar.Schema.from_dict(NumpyField.schema)
        # deserial = schema().load(value)
        deserial = value

        return np.array(deserial["data"], dtype=np.dtype(deserial["dtype"]))
