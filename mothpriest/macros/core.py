from ..parsers import IntegerParser
from functools import partial

# unsigned integer sizes
uint8 = partial(IntegerParser, 1)
uint16 = partial(IntegerParser, 2)
uint32 = partial(IntegerParser, 4)