"""Marshmallow numpy field."""
# @Author: George Onoufriou <archer>
# @Date:   2021-08-20T16:38:06+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-20T17:00:45+01:00

import marshmallow as mar


class Numpy(mar.fields.Field):
    """Marshmallow field to serialise and deserialise numpy data."""
