from abc import ABC, abstractmethod
import struct
from collections import OrderedDict
from typing import Union, List, Callable
from functools import partial
from io import BytesIO
from PIL import Image
from ..references import *

class Parser(ABC):
    """Abstract base class for parser objects"""

    def __init__(self, id: str):
        self.id = id
        self._parent = None
        self._record = None

    def getID(self):
        """Returns the id of this element"""
        return self.id

    @abstractmethod
    def parse(self, buffer: BytesIO):
        """Parse input bytes"""
        pass

    @abstractmethod
    def unparse(self, buffer: BytesIO):
        """Unparse record back into bytes"""
        pass

    @abstractmethod
    def getSize(self):
        """get size of this element"""
        pass

    def getReference(self, reference: Union[Reference, str, int, List[str]]):
        """allows the use of strings and lists of strings in place of the corresponding Reference subclasses"""
        ref = Reference.fromOther(reference)
        return ref.retrieveRecord(self)

class FixedSizeParser(Parser):
    """Abstract subclass for parsing a fixed number of bytes"""

    def __init__(self, id: str, size: int):
        """Basic initializer"""
        super().__init__(id)
        self.size = size

    def getSize(self):
        """Get the constant, fixed size of this element"""
        return self.size

class FixedSizeRawParser(FixedSizeParser):

    def parse(self, buffer: BytesIO):
        self._record = buffer.read(self.size)
    
    def unparse(self, buffer: BytesIO):
        buffer.write(self._record)

# class FixedBlockParser(FixedSizeParser):
#     """Class for parsing a block of parsable elements each having a fixed size"""

#     def __init__(self, elements: List[FixedSizeParser], id: str):
#         """Pass a list of FixedSizeParser objects and an id"""
        
#         size = 0
#         ids = set()
#         for element in elements:
#             if element.id in ids:
#                 raise ValueError('Duplicate ID found')
#             ids.add(element.id)
#             size += element.size
#         super().__init__(size)
#         self.elements = elements
#         self.id = id

#     def parse(self, data: bytes):
#         self.checkSize(data)
#         offset = 0
#         record = {}
#         for element in self.elements:
#             if element.id is not None:
#                 record[element.id] = element.parse(data[offset:offset+element.size])
#             else:
#                 element.parse(data[offset:offset+element.size])
#             offset += element.size
#         return record

class MagicParser(FixedSizeParser):
    """Class for parsing magic strings, which usually exist at the beginning of a file to hint at its file type"""
    
    def __init__(self, value: Union[str, bytes], id: str):
        """pass a size, the expected magic value, and an id"""
        size = len(value)
        super().__init__(id, size)
        if isinstance(value, bytes):
            value = value
        elif isinstance(value, str):
            value = value.encode('utf-8')
        else:
            raise ValueError('expected value to be either bytes or str')
        super().__init__(id, len(value))
        self.value = value

    def parse(self, buffer: BytesIO):
        """Raises a ValueError if the input does not exactly match the expected magic string"""

        data = buffer.read(self.getSize())
        if not data == self.value:
            raise ValueError('Magic did not match expected value. Check your parser and that you passed the correct file')
        self._record = data

    def unparse(self, buffer: BytesIO):
        buffer.write(self.value)

class StructValueParser(FixedSizeParser):
    """Class for parsing a packed value using struct.unpack"""
    
    def __init__(self, size: int, format_string: str, id: str):
        """Pass a size, a format string for the struct.unpack method, and an id"""
        super().__init__(id, size)
        self.format_string = format_string

    def parse(self, buffer: BytesIO):
        """returns a list of unpacked values"""
        data = buffer.read(self.size)
        self._record = struct.unpack(self.format_string, data)
    
    def unparse(self, buffer: BytesIO):
        try:
            packed = struct.pack(self.format_string, *self._record)
        except TypeError:
            packed = struct.pack(self.format_string, self._record)
        buffer.write(packed)
    
class IntegerParser(StructValueParser):
    """Class for parsing packed integers with a given size"""
    mapping = {
        1: 'b',
        2: 'h',
        4: 'i'
    }

    def __init__(self, size: int, id: str, little_endian: bool = True, signed=False):
        """Pass a valid size (number of bytes), endianness, and whether or not the value is signed or unsigned"""
        if not size in self.mapping.keys():
            raise ValueError('Size not supported')
        
        format_string = self.mapping[size]
        if not signed:
            format_string = format_string.upper()

        if little_endian:
            format_string = '<' + format_string
        else:
            format_string = '>' + format_string
        
        super().__init__(size, format_string, id)

    def parse(self, buffer: BytesIO):
        """returns a parsed integer"""
        super().parse(buffer)
        values = self._record
        try:
            assert len(values) == 1
            assert isinstance(values[0], int)
        except:
            raise ValueError(f'failed to parse integer, got {values}')
        self._record = values[0]

class ReferenceSizeParser(Parser):
    """Abstract subclass for parsing a variable number of bytes given by a reference"""

    def __init__(self, id: str, size_id: Reference):
        super().__init__(id)
        self.size_id = size_id

    def getSize(self):
        """Retrieve a size from the record for already-parsed input"""
        if self._parent is None:
            raise ValueError(f'Failed to get size for object with id <{self.getID()}>, object has no parent')
        return self._parent.getReference(self.size_id)._record
        
class ReferenceSizeStringParser(ReferenceSizeParser):
    """Class for parsing reference-sized strings"""

    def parse(self, buffer: BytesIO):
        value = buffer.read(self.getSize())
        self._record = value.decode('utf-8')
    
    def unparse(self, buffer: BytesIO):
        buffer.write(self._record.encode('utf-8'))

class ReferenceSizeImageParser(Parser):
    """Class for parsing reference-sized image data"""

    def __init__(self, width_id: Reference, height_id: Reference, id: str, depth: Reference = 1, numColors: Reference = 4):
        """Create an image parser where depth is given in bytes and numColors is an integer referring to RGB vs RGBA, etc."""
        super().__init__(id)
        self.width_id = width_id
        self.height_id = height_id
        self.depth = depth
        self.numColors = numColors

    def _getDims(self):
        width = self._parent.getReference(self.width_id)._record
        height = self._parent.getReference(self.height_id)._record
        if isinstance(self.depth, int):
            depth = self.depth
        else:
            depth = self._parent.getReference(self._record, self.depth)._record
        return (width, height, depth)

    def getSize(self):
        width, height, depth = self._getDims()
        if isinstance(self.numColors, int):
            numColors = self.numColors
        else:
            numColors = self._parent.getReference(self.depth)._record
        return width*height*depth*numColors

    def _getMode(self):
        """Get Pillow mode"""
        if self.depth != 1:
            raise ValueError('Currently only a byte-depth of 1 is supported')
        colorsToMode = {
            1: 'L',
            3: 'RGB',
            4: 'RGBA'
        }
        try:
            return colorsToMode[self.numColors]
        except KeyError:
            raise ValueError(f'Unsupported number of colors {self.numColors}')
    
    def parse(self, buffer: BytesIO):
        image_data = buffer.read(self.getSize())
        width, height, depth = self._getDims()
        mode = self._getMode()
        image = Image.frombytes(mode, (width, height), image_data)
        self._record = image
    
    def unparse(self, buffer: BytesIO):
        image_bytes = self._record.tobytes()
        buffer.write(image_bytes)

class BlockParser(Parser):
    """Class for parsing a block of parsable elements"""

    def __init__(self, id: str, elements: List[Union[FixedSizeParser, ReferenceSizeParser]]):
        super().__init__(id)
        self._record = OrderedDict()
        for element in elements:
            self._record[element.getID()] = element
            element._parent = self

    def getSize(self):
        total = 0
        for id, element in self._record.items():
            size = element.getSize()
            if size is None:
                return None
            total += size
        return total

    def parse(self, buffer: BytesIO):
        for id, element in self._record.items():
            element.parse(buffer)
    
    def unparse(self, buffer: BytesIO):
        for id, element in self._record.items():
            element.unparse(buffer)

class ReferenceSizeBlockParser(ReferenceSizeParser, BlockParser):
    """Class for parsing a block of elements which has a referenced size"""

    def __init__(self, id: str, size_id: Reference, elements: List[Parser]):
        self.id = id
        self.size_id = size_id
        self._parent = None
        self._record = OrderedDict()
        for element in elements:
            self._record[element.getID()] = element
            element._parent = self

class RawParser(Parser):

    def parse(self, buffer: BytesIO):
        return buffer.read(self.getSize())
    
    def getSize(self):
        if self.id not in self._record:
            return None
        return len(self.getReference(self._record, self.id))

    def unparse(self, buffer: BytesIO):
        buffer.write(self._record)
    
class ReferenceSizeRawParser(ReferenceSizeParser, RawParser):
    pass
    
class ReferenceCountParser(Parser):

    def __init__(self, id: str, count_id: Reference, element_factory: Callable[[int], Parser]):
        super().__init__(id)
        self._record = []
        self.count_id = count_id
        self.element_factory = element_factory
        self._count = -1

    def getSize(self):
        if self._count < 0:
            return None
        if len(self._record) != self._count:
            raise ValueError(f'length of self._record ({len(self._record)}) must be equal to self._count ({self._count})')
        total = 0
        for element in self._record:
            size = element.getSize()
            total += size
        return total

    def parse(self, buffer: BytesIO):
        self._count = self._parent.getReference(self.count_id)._record
        for i in range(0, self._count):
            self._record.append(self.element_factory(i))
            self._record[i]._parent = self
            self._record[i].parse(buffer)

    def unparse(self, buffer: BytesIO):
        for element in self._record:
            element.unparse(buffer)
    
class TransformationParser(ReferenceSizeBlockParser):

    def __init__(self, id: str, size_id: str, transform, transformInverse, elements: List[Parser]):
        super().__init__(id, size_id, elements)
        self.transform = transform
        self.transformInverse = transformInverse

    def parse(self, buffer: BytesIO):
        size = self.getSize()
        data = buffer.read(size)
        transformedBuffer = BytesIO(self.transform(data))
        super().parse(transformedBuffer)
    
    def unparse(self, buffer: BytesIO):
        tempbuf = BytesIO()
        self._record.unparse(tempbuf)
        tempbuf.seek(0)
        buffer.write(self.transformInverse(tempbuf.read()))
    
class FixedPaddingParser(FixedSizeParser):

    def __init__(self, size: int):
        super().__init__(None, size)

    def parse(self, buffer: BytesIO):
        self._record = buffer.read(self.getSize())

    def unparse(self, buffer: BytesIO):
        new_padding = b'\x00' * self.size
        buffer.write(new_padding)

class BytesExpansionParser(TransformationParser):
    """Class for expanding bytes to bits. Every byte gets expanded to 8 bytes each containing either x00 or x01.
    This is space inefficient but means that existing parser objects can be used on 'bits' as well"""

    def _expand(self, data: bytes, bit_sizes: List[int]):
        result = b''
        bits = ''.join([format(byte, '08b') for byte in data])
        for bit_size in bit_sizes:
            # import pdb; pdb.set_trace()
            current_bits = bits[:bit_size]
            bits = bits[bit_size:]
            offset = (8 - bit_size) % 8
            for i in range(-offset, bit_size-offset, 8):
                byte_value = int(current_bits[max(i, 0):i+8], 2)
                result += struct.pack('B', byte_value)
        return result
    
    def _contract(self, data: bytes, bit_sizes: List[int]):
        bits = ''
        data_array = bytearray(data)
        for bit_size in bit_sizes:
            remainder = bit_size % 8
            if remainder != 0:
                bits += format(data_array.pop(0), f'0{remainder}b')
            for _ in range(0, bit_size // 8):
                bits += format(data_array.pop(0), '08b')
        assert len(bits) / 8 == self.size
        result = b''
        for i in range(0, self.size):
            result += struct.pack('B', int(bits[i*8:i*8+8], 2))
        return result
                

    def __init__(self, id: str, size: int, bit_sizes: List[int], parser: FixedSizeParser):
        """Note that size here indicates the size in bytes, not bits"""
        if size != sum(bit_sizes) / 8:
            raise ValueError('Number of bits must add up to 1/8th the number of bytes')
        super().__init__(id, ConstIntegerReference(size), partial(self._expand, bit_sizes=bit_sizes), partial(self._contract, bit_sizes=bit_sizes), [parser])
    
class PDBParser(Parser):

    def getSize(self):
        return None
    
    def parse(self, buffer: BytesIO):
        import pdb; pdb.set_trace()
        pass

    def unparse(self, buffer: BytesIO):
        raise NotImplementedError('unparsing not implemented for debug parsers')
    
class ReferenceMappedParser(Parser):

    def __init__(self, id: str, key_id: Reference, mapping: dict):
        super().__init__(id)
        self.key_id = key_id
        self.mapping = mapping
        self._active = None

    def _getParser(self):
        key = self._parent.getReference(self.key_id)._record
        try:
            self._active = self.mapping[key]
            self._active._parent = self._parent
        except KeyError:
            raise KeyError(f'ReferenceMappedParser read an unexpected key: {key}')

    def getSize(self):
        self._getParser()
        return self._active.getSize()
    
    def parse(self, buffer: BytesIO):
        self._getParser()
        self._active.parse(buffer)
        self._record = self._active._record

    def unparse(self, buffer: BytesIO):
        self._getParser()
        self._active._record = self._record
        self._active.unparse(buffer)
    
class ReferenceSizeChunkParser(ReferenceSizeParser):
    """Class for chunking the input bytes to allow for chunks to only be partially read if their size is known ahead of time"""

    def __init__(self, id: str, size_id: Reference, parser: Parser):
        super().__init__(id, size_id)
        self.parser = parser

    def getID(self):
        return self.parser.getID()

    def parse(self, buffer: BytesIO):
        chunk = BytesIO(buffer.read(self.getSize()))
        self.parser._parent = self._parent
        self.parser.parse(chunk)
        self._record = self.parser._record
    
    def unparse(self, buffer: BytesIO):
        self.parser._record = self._record
        self.parser.unparse(buffer)
    
class EOFParser(Parser):

    def getSize(self):
        return None
    
    def __init__(self):
        super().__init__('_eof')

    def parse(self, buffer: BytesIO):
        remainder = buffer.read()
        if len(remainder) != 0:
            raise ValueError(f'Expected eof but found {len(remainder)} bytes instead')
    
    def unparse(self, buffer: BytesIO):
        pass
    
class FixedSizeHexParser(FixedSizeParser):

    def __init__(self, id: str, size: int, little_endian=True):
        super().__init__(id, size)
        self.little_endian = little_endian

    def parse(self, buffer: BytesIO):
        content = buffer.read(self.getSize())
        hex_bytes = [format(byte, '02X') for byte in content]
        if self.little_endian:
            hex_bytes = reversed(hex_bytes)
        return ''.join(hex_bytes)
        
    def unparse(self, buffer: BytesIO):
        record = self._record[self.getID()]
        hex_bytes = bytes.fromhex(record)
        if self.little_endian:
            hex_bytes = bytes(reversed(hex_bytes))
        buffer.write(hex_bytes)

# TODO implement fixed size as a constant value reference (refactor)
