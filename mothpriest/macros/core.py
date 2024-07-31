from ..parsers import IntegerParser, StructValueParser
from functools import partial

# unsigned integer sizes
uint8 = partial(IntegerParser, size=1)
uint16 = partial(IntegerParser, size=2)
uint32 = partial(IntegerParser, size=4)

# floating point
float32 = lambda id: StructValueParser(4, 'f', id)

float16 = lambda id: StructValueParser(2, 'e', id)