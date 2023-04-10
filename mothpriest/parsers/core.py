from abc import ABC, abstractmethod
import struct
from typing import Union, List
from functools import partial
from io import BytesIO
from PIL import Image
from ..references import *

class Parser(ABC):
    """Abstract base class for parser objects"""

    def __init__(self, id: str):
        self.id = id
        self._record = {}

    def updateRecord(self, record):
        """Updates internal storage of the records that have been parsed"""
        self._record = record

    def getId(self):
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

    def getReference(self, record, reference: Union[Reference, str, int, List[str]]):
        """allows the use of strings and lists of strings in place of the corresponding Reference subclasses"""
        if isinstance(reference, Reference):
            ref = reference
        elif isinstance(reference, str):
            ref = IDReference(reference)
        elif isinstance(reference, int):
            ref = ConstIntegerReference(reference)
        elif isinstance(reference, list):
            ref = IDListReference(reference)
        else:
            import pdb; pdb.set_trace()
        return ref.retrieveRecord(record)

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
        return buffer.read(self.size)
    
    def unparse(self, buffer: BytesIO):
        buffer.write(self._record[self.id])

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

        data = buffer.read(self.size)
        if not data == self.value:
            raise ValueError('Magic did not match expected value. Check your parser and that you passed the correct file')
        return self.value
    
    def unparse(self, buffer: BytesIO):
        buffer.write(self.value)

class StructValueParser(FixedSizeParser):
    """Class for parsing a packed value using struct.unpack"""
    
    def __init__(self, size: int, unpack_string: str, id: str):
        """Pass a size, a format string for the struct.unpack method, and an id"""
        self.size = size
        self.unpack_string = unpack_string
        self.id = id

    #@profile
    def parse(self, buffer: BytesIO):
        data = buffer.read(self.size)
        return struct.unpack(self.unpack_string, data)
    
    def unparse(self, buffer: BytesIO):
        buffer.write(struct.pack(self.unpack_string, *self._record))
    
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
        return super().parse(buffer)[0]
    
    def unparse(self, buffer: BytesIO):
        super().unparse([self._record], buffer)

class ReferenceSizeParser(Parser):
    """Abstract subclass for parsing a variable number of bytes given by a reference"""

    def __init__(self, id: str, size_id: Reference):
        super().__init__(id)
        self.size_id = size_id

    def getSize(self):
        """Retrieve a size from the record for already-parsed input"""
        assert self._record is not None
        return self.getReference(self._record, self.size_id)
        
class ReferenceSizeStringParser(ReferenceSizeParser):
    """Class for parsing reference-sized strings"""

    def parse(self, buffer: BytesIO):
        value = buffer.read(self.getSize())
        return value.decode('utf-8')
    
    def unparse(self, buffer: BytesIO):
        buffer.write(self._record.encode('utf-8'))
    
# TODO subclass from a matrix or tensor parser?
class ReferenceSizeImageParser(ReferenceSizeParser):
    """Class for parsing reference-sized image data"""

    def __init__(self, width_id: Reference, height_id: Reference, id: str, depth: Union[int, Reference] = 1, numColors: Union[int, Reference] = 4):
        self.width_id = width_id
        self.height_id = height_id
        self.id = id
        self._record = {}
        self.depth = depth #TODO convert to bits
        self.numColors = numColors

    def _getDims(self):
        width = self.getReference(self._record, self.width_id)
        height = self.getReference(self._record, self.height_id)
        if isinstance(self.depth, int):
            depth = self.depth
        else:
            depth = self.getReference(self._record, self.depth)
        return (width, height, depth)

    def getSize(self):
        width, height, depth = self._getDims()
        if isinstance(self.numColors, int):
            numColors = self.numColors
        else:
            numColors = self.getReference(self._record, self.depth)

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
        return image
    
    def unparse(self, buffer: BytesIO):
        image = self._record
        image_bytes = image.tobytes()
        buffer.write(image_bytes)

class BlockParser(Parser):
    """Class for parsing a block of parsable elements"""

    def __init__(self, id: str, elements: List[Union[FixedSizeParser, ReferenceSizeParser]]):
        super().__init__(id)
        self.elements = elements

    #@profile
    def getSize(self):
        try:
            record = self.getReference(self._record, self.id)
        except (KeyError, IndexError) as e:
            return None

        total = 0
        for element in self.elements:
            element.updateRecord(record)
            size = element.getSize()
            if size is None:
                return None
            total += size
        return total

    def parse(self, buffer: BytesIO):
        record = {'_parent': self._record}
        for element in self.elements:
            element.updateRecord(record)
            record[element.getId()] = element.parse(buffer)
        del record['_parent']
        return record
    
    def unparse(self, buffer: BytesIO):
        for element in self.elements:
            if element.id is not None:
                element.unparse(self._record[element.id], buffer)
            else:
                element.unparse(self._record, buffer)

class ReferenceSizeBlockParser(ReferenceSizeParser, BlockParser):
    """Class for parsing a block of elements which has a referenced size"""

    def __init__(self, id: str, size_id: Reference, elements: List[Parser]):
        self.id = id
        self.size_id = size_id
        self.elements = elements
        self._record = {}

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
    
    def __init__(self, id: str, count_id: Reference, element_parser: Parser):
        super().__init__(id)
        self.count_id = count_id
        self.element_parser = element_parser
        self._count = -1
        self._range = None

    def updateRecord(self, record):
        super().updateRecord(record)
        self._count = self.getReference(self._record, self.count_id)
        self._range = range(0, self._count)

    def getSize(self):
        if self.id not in self._record:
            return None
        record = self._record[self.id]

        total = 0
        ep = self.element_parser
        ep.updateRecord(record)
        for i in self._range:
            ep.id = i
            total += ep.getSize()
        return total

    def parse(self, buffer: BytesIO):
        record = []
        ep = self.element_parser
        ep.updateRecord(record)
        try:
            for _ in self._range:
                ep.updateRecord(record)
                element_record = ep.parse(buffer)
                record.append(element_record)
        except TypeError:
            import pdb; pdb.set_trace()
        return record

    def unparse(self, buffer: BytesIO):
        id_base = self.element_parser.id
        for i in range(0, self._count):
            self.element_parser.id = id_base + f'_{i}'
            if self.element_parser.id is not None:
                self.element_parser.unparse(self._record[self.element_parser.id])
            else:
                self.element_parser.unparse(self._record)
        self.element_parser.id = id_base

class DebugRemainderParser(FixedSizeParser):

    def __init__(self, size: int):
        super().__init__('debug', size)

    def parse(self, buffer: BytesIO):
        return buffer.read(self.size)

    def getSize(self):
        return None
    
    def unparse(self):
        raise NotImplementedError('unparsing not supported for debugging parsers')
    
class TransformationParser(ReferenceSizeParser):

    def __init__(self, id: str, size_id: str, transform, transformInverse, parser: Parser):
        super().__init__(id, size_id)
        self.transform = transform
        self.transformInverse = transformInverse
        self.parser = parser

    def parse(self, buffer: BytesIO):
        size = self.getSize()
        data = buffer.read(size)
        transformed = BytesIO(self.transform(data))
        return self.parser.parse(transformed)
    
    def unparse(self, buffer: BytesIO):
        tempbuf = BytesIO()
        self.parser.updateRecord(self._record[self.parser.getId()])
        self.parser.unparse(tempbuf)
        tempbuf.seek(0)
        buffer.write(self.transformInverse(tempbuf.read()))
    
class FixedPaddingParser(FixedSizeParser):

    def __init__(self, size: int):
        super().__init__(None, size)

    def parse(self, buffer: BytesIO):
        return buffer.read(self.getSize())

    def unparse(self, buffer: BytesIO):
        new_padding = b'\x00' * self.size
        buffer.write(new_padding)

class BytesExpansionParser(FixedSizeParser, TransformationParser):
    """Class for expanding bytes to bits. Every byte gets expanded to 8 bytes each containing either x00 or x01.
    This is space inefficient but means that existing parser objects can be used on 'bits' as well"""

    def _expand(self, data: bytes, bit_sizes: List[int]):
        result = b''
        bits = ''.join([format(byte, '08b') for byte in data])
        for bit_size in bit_sizes:
            # import pdb; pdb.set_trace()
            current_bits = bits[:bit_size]
            bits = bits[bit_size:]
            offset = 8 - bit_size % 8
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

        self.id = id
        self.size = size
        self._record = {}
        self.transform = partial(self._expand, bit_sizes=bit_sizes)
        self.transformInverse = partial(self._contract, bit_sizes=bit_sizes)
        self.parser = parser

    def getSize(self):
        return super().getSize()
    
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
        key = self.getReference(self._record, self.key_id)
        try:
            return self.mapping[key]
        except KeyError:
            raise KeyError(f'ReferenceMappedParser read an unexpected key: {key}')

    def updateRecord(self, record):
        self._record = record
        self._active = self._getParser()
        self._active.updateRecord(record)

    def getSize(self):
        return self._active.getSize()
    
    def parse(self, data: bytes):
        return self._active.parse(data)

    def unparse(self, buffer: BytesIO):
        self._active.unparse(self._record, buffer)
    
class ReferenceSizeChunkParser(ReferenceSizeParser):
    """Class for chunking the input bytes to allow for chunks to only be partially read if their size is known ahead of time"""

    def __init__(self, id: str, size_id: Reference, parser: Parser):
        super().__init__(id, size_id)
        self.parser = parser

    def parse(self, buffer: BytesIO):
        chunk = BytesIO(buffer.read(self.getSize()))
        self.parser.updateRecord(self._record)
        return self.parser.parse(chunk)
    
    def unparse(self, buffer: BytesIO):
        raise NotImplementedError('unparsing not yet implemented')
    
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
        hex_bytes = bytes.fromhex(self._record)
        if self.little_endian:
            hex_bytes = reversed(hex_bytes)
        buffer.write(hex_bytes)

# TODO implement fixed size as a constant value reference (refactor)
