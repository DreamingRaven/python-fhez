# @Author: GeorgeRaven <archer>
# @Date:   2021-02-22T11:46:18+00:00
# @Last modified by:   archer
# @Last modified time: 2021-03-11T12:55:27+00:00
# @License: please see LICENSE file in project root
from fhe.nn.block.block import Block


class Activation(Block):

    @property
    def is_activation(self):
        """Are we an Activation function."""
        return True

    @property
    def is_layer(self):
        """Are we a Layer."""
        return False
