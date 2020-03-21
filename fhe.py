#!/usr/bin/env python3

# @Author: GeorgeRaven <archer>
# @Date:   2020-03-21T11:30:56+00:00
# @Last modified by:   archer
# @Last modified time: 2020-03-21T11:44:36+00:00
# @License: please see LICENSE file in project root

import os


class Fhe(object):
    """Fully Homomorphic Encryption (FHE) utility library.

    This library is designed to streamline and simplify FHE encryption,
    decryption, abstraction, serialisation, and integration. In particular
    this library is intended for use in deep learning to fascilitate a
    more private echosystem.


    :param args: Dictionary of overides.
    :param logger: Function address to print/ log to (default: print).
    :type args: dictionary
    :type logger: function address
    :example: Fhe()
    """

    def __init__(self, args=None, logger=None):
        """Init class with defaults.

        optionally accepts dictionary of default and logger overrides.
        """
        args = args if args is not None else dict()
        self.home = os.path.expanduser("~")
        defaults = {

            "pylog": logger if logger is not None else print,

        }
        self.args = self._mergeDicts(defaults, args)
        # final adjustments to newly defined dictionary
        pass

    def _mergeDicts(self, *dicts):
        """Given multiple dictionaries, merge together in order."""
        result = {}
        for dictionary in dicts:
            result.update(dictionary)  # merge each dictionary in order
        return result

    def debug(self):
        """Display current internal state."""
        self.args["pylog"](self.args)


if __name__ == "__main__":
    fhe = Fhe()
    fhe.debug()
