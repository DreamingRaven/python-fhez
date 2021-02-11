# @Author: GeorgeRaven <archer>
# @Date:   2021-02-11T11:33:12+00:00
# @Last modified by:   archer
# @Last modified time: 2021-02-11T11:33:41+00:00
# @License: please see LICENSE file in project root

import marshmallow


class ReScheme(marshmallow.Schema):
    """Marshmallow serialisation schema.

    This schema works with ReSeal to help in transmission of serialised ReSeal
    objects. This also helps to verify the contents are structured as expected
    on the recieving end. However since byte strings are encoded as strings
    there is little further testing that can be done on them.
    """
    _scheme = marshmallow.fields.Integer()
    _poly_modulus_degree = marshmallow.fields.Integer()
    _coefficient_modulus = marshmallow.fields.List(
        marshmallow.fields.Integer())
    _scale = marshmallow.fields.Float()
    _parameters = marshmallow.fields.Dict(keys=marshmallow.fields.Str(),
                                          values=marshmallow.fields.Str())
    _public_key = marshmallow.fields.Dict(keys=marshmallow.fields.Str(),
                                          values=marshmallow.fields.Str())
    _private_key = marshmallow.fields.Dict(keys=marshmallow.fields.Str(),
                                           values=marshmallow.fields.Str())
    _relin_keys = marshmallow.fields.Dict(keys=marshmallow.fields.Str(),
                                          values=marshmallow.fields.Str())
    _ciphertext = marshmallow.fields.Dict(keys=marshmallow.fields.Str(),
                                          values=marshmallow.fields.Str())
