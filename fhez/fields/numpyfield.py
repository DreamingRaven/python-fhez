"""Marshmallow numpy field."""
# @Author: George Onoufriou <archer>
# @Date:   2021-08-20T16:38:06+01:00
# @Last modified by:   archer
# @Last modified time: 2021-09-09T14:24:33+01:00

import numpy as np
import marshmallow as mar


class NumpyField(mar.fields.Field):
    """Marshmallow field to serialise and deserialise numpy data."""

    @property
    def schema(self):
        """Get numpy field marshamllow schema."""
        schema = {
            "dtype": mar.fields.Str(),
            "dshape": mar.fields.List(mar.fields.Int()),
            "data": mar.fields.List(mar.fields.Float())
        }
        return mar.Schema.from_dict(schema)

    def _serialize(self, value, attr, obj, **kwargs):
        """Use marshmallow to serialise numpy as list of numbers."""
        if value is None:
            return None

        # define and set the to be stored data
        store = {
            "dtype": value.dtype.name,
            "dshape": list(value.shape),
            "data": value.flatten().tolist()
        }
        # now used marshmallow existing handler desired
        serialised = self.schema().dump(store)
        return serialised

    def _deserialize(self, value, attr, data, **kwargs):
        """Use marshmallow to deserialise list of numbers back to numpy."""
        if value is None:
            return None

        # schema = mar.Schema.from_dict(NumpyField.schema)
        deserial = self.schema().load(value)
        # deserial = value

        return np.array(deserial["data"],
                        dtype=np.dtype(deserial["dtype"])
                        ).reshape(tuple(deserial["dshape"]))
