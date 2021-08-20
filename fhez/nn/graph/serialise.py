"""Serialisation base abstraction to unify/ standardise nodes."""
# @Author: George Onoufriou <archer>
# @Date:   2021-08-20T13:50:11+01:00
# @Last modified by:   archer
# @Last modified time: 2021-08-20T13:56:24+01:00
import abc
import marshmallow as mar


class Serialise(abc.ABC):
    """Abstract base class to standardise serialisation across all nodes."""

    @property
    @abc.abstractmethod
    def schema(self):
        """Get Marshmallow schema for this class for (de)serialisation."""

    def __getstate__(self):
        """Get current state in basic inbuilt-objects for serialisation."""
        schema = self.schema
        serialised = schema().dump(self)
        return serialised

    def __setstate__(self, d):
        """Set the current state of the class using input dict repr."""
        schema = self.schema
        deserialised = schema().load(d)
        self.__dict__ = deserialised

    def __eq__(self, other):
        """Check equality by comparison of internal dictionary."""
        if self.__class__ != other.__class__:
            return False
        return self.__dict__ == other.__dict__
