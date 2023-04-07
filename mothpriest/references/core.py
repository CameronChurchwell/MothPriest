from abc import ABC, abstractmethod
from typing import List

class Reference(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def retrieveRecord(self, record):
        pass


class IDReference(Reference):
    """Class implementing ID references as strings for single level lookup"""

    def __init__(self, id: str):
        self.id = id

    def retrieveRecord(self, record):
        """retrieve the value corresponding to this reference from a given record"""
        try:
            return record[self.id]
        except (KeyError, IndexError):
            raise ValueError(f'failed to find reference {self.id} in record')
        
class IDListReference(Reference):
    """Class implementing ID references which traverse levels in the record"""

    def __init__(self, references: List[IDReference]):
        self.references = references

    #TODO consider generalizing
    def retrieveRecord(self, record):
        rec = record
        for ref in self.references:
            rec = rec[ref]
        return rec
        
class ConstIntegerReference(Reference):
    """Class implementing references as a fixed constant integer"""

    def __init__(self, value: int):
        self.value = value

    def retrieveRecord(self, record):
        return self.value
    
class FunctionReference(Reference):
    """Class implementing references as a function of one or more other references"""

    def __init__(self, function, references: List[Reference]):
        self.function = function
        self.references = references

    def retrieveRecord(self, record):
        values = [ref.retrieveRecord(record) for ref in self.references]
        return self.function(*values)
    
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

    def retrieveRecord(self, record):
        try:
            return self.primary.retrieveRecord(record)
        except (KeyError, IndexError):
            return self.fallback.retrieveRecord(record)
        
class FallbackChainReference(FallbackReference):
    """Class implementing a chain of fallback references (essentially a linked list)"""

    def __init__(self, references: List[Reference]):
        super().__init__(references[0], FallbackChainReference(references[1:]))