from __future__ import annotations
from typing import List, Union, TYPE_CHECKING
if TYPE_CHECKING:
    from ..parsers import BlockParser, ReferenceCountParser

from .. import parsers

from abc import ABC, abstractmethod

class Reference(ABC):

    @abstractmethod
    def __init__(self):
        raise NotImplementedError()

    @abstractmethod
    def retrieveRecord(self, record):
        raise NotImplementedError()

    @classmethod
    def fromOther(cls, ref):
        if isinstance(ref, Reference):
            return ref
        if isinstance(ref, str):
            return IDReference(ref)
        elif isinstance(ref, int):
            return ConstIntegerReference(ref)
        elif isinstance(ref, list) or isinstance(ref, tuple):
            return IDListReference([IDReference(s) for s in ref])
        else:
            raise ValueError(f'Unknown alias {ref}')

class IDReference(Reference):
    """Class implementing ID references as strings for single level lookup"""

    def __init__(self, id: Union[str, int]):
        self.id = id

    def retrieveRecord(self, parser: Union[BlockParser, ReferenceCountParser]):
        """retrieve the value corresponding to this reference from a given record"""
        if self.id == '_parent':
            return parser._parent
        try:
            return parser._record[self.id]
        except (KeyError, IndexError):
            raise ValueError(f'failed to find reference {self.id} in record {parser._record}')
        except TypeError:
            import pdb; pdb.set_trace()
        
class IDListReference(Reference):
    """Class implementing ID references which traverse levels in the record"""

    def __init__(self, references: List[IDReference]):
        self.references = references

    def retrieveRecord(self, parser: Union[BlockParser, ReferenceCountParser]):
        p = parser
        for ref in self.references:
            try:
                p = p.getReference(ref)
            except (KeyError, IndexError):
                raise ValueError(f'cannot retrieve reference {ref} from parser {p} with id {p.getID()}')
        return p
        
class ConstIntegerReference(Reference):
    """Class implementing references as a fixed constant integer"""

    def __init__(self, value: int):
        self._record = value

    def getRecord(self):
        return self._record

    def retrieveRecord(self, parser):
        return self
    
class FunctionReference(Reference):
    """Class implementing references as a function of one or more other references"""

    def __init__(self, function, references: List[Reference]):
        self.function = function
        self.references = references
        self._record = None

    def retrieveRecord(self, parser):
        values = [ref.retrieveRecord(parser)._record for ref in self.references]
        self._record = self.function(*values)
        return self
    
    def getRecord(self):
        return self._record

class SumReference(FunctionReference):
    """Class implementing references as a sum"""

    def __init__(self, references: List[Reference]):
        super().__init__(sum, references)

class NegationReference(FunctionReference):
    """Class implementing references as a negation"""

    def __init__(self, reference: Reference):
        super().__init__(lambda x: -x, reference)

class DifferenceReference(FunctionReference):
    """Class implementing references as a difference between two other references"""

    def __init__(self, first: Reference, second: Reference):
        super().__init__(lambda x, y: x-y, [first, second])
    
class FallbackReference(Reference):
    """Class implementing a fallback reference in the case of a key or index error"""

    def __init__(self, primary: Reference, fallback: Reference):
        self.primary = primary
        self.fallback = fallback

    def retrieveRecord(self, parser):
        try:
            return self.primary.retrieveRecord(parser)
        except (KeyError, IndexError):
            return self.fallback.retrieveRecord(parser)
        
class FallbackChainReference(FallbackReference):
    """Class implementing a chain of fallback references (essentially a linked list)"""

    def __init__(self, references: List[Reference]):
        super().__init__(references[0], FallbackChainReference(references[1:]))

extendedReference = Union[Reference, str, int, List[str], None]