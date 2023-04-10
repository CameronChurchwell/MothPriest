from ..parsers import IntegerParser, StructValueParser
from functools import partial

# unsigned integer sizes
uint8 = partial(IntegerParser, 1)
uint16 = partial(IntegerParser, 2)
uint32 = partial(IntegerParser, 4)

# floating point
float32 = lambda id: StructValueParser(4, 'f', id)