from abc import ABC, abstractmethod
import struct
from collections import OrderedDict
from typing import Union, List, Callable, Dict
from functools import partial
from io import BytesIO
from PIL import Image
import json
from ..references import *

class Parser(ABC):
    """Abstract base class for parser objects"""

    def __init__(self, id: str):
        self.id = id
        self._parent = None
        self._record = None
        self._size = None

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

    def getAllRecords(self):
        return self

    def __str__(self):
        """get a string representation of the record stored in this parser"""
        return str(self._record)
    
    def __call__(self, buffer: BytesIO):
        self.parse(buffer)

    def getReference(self, reference: Union[Reference, str, int, List[str]]):
        """allows the use of strings and lists of strings in place of the corresponding Reference subclasses"""
        ref = Reference.fromOther(reference)
        return ref.retrieveRecord(self)
    
    def _updateParent(self):
        if self._parent is not None:
            self._parent.updateRecord()

    def updateRecord(self, value=None):
        """Update the record stored in this parser with a new value"""
        if value is not None:
            self._record = value
        self._updateParent()

    def _getInternalSize(self):
        """get the size currently store in self._record, which may not be the same value as self.getSize()"""
        dummy_stream = BytesIO()
        self.unparse(dummy_stream)
        new_size = dummy_stream.tell()
        del dummy_stream
        return new_size

class FixedSizeParser(Parser):
    """Abstract subclass for parsing a fixed number of bytes"""

    def __init__(self, id: str, size: int):
        """Basic initializer"""
        super().__init__(id)
        self.size = size

    def getSize(self):
        """Get the constant, fixed size of this element"""
        return self.size
    
    def updateRecord(self, value=None):
        if value is not None:
            new_size = len(value)
            if new_size != self.size:
                raise ValueError(f'new size {new_size} does not match fixed size {self.size} for parser {self.id}')
        self._updateParent()

    def _getInternalSize(self):
        return self.size

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
        self._record = value

    def parse(self, buffer: BytesIO):
        """Raises a ValueError if the input does not exactly match the expected magic string"""

        data = buffer.read(self.getSize())
        if not data == self._record:
            raise ValueError('Magic did not match expected value. Check your parser and that you passed the correct file')

    def unparse(self, buffer: BytesIO):
        buffer.write(self._record)

    def __str__(self):
        return str(self._record)
    
    def getAllRecords(self):
        return str(self._record)

    def updateRecord(self, value=None):
        raise NotImplementedError('Cannot update magic record, operation is not permitted')

class StructValueParser(FixedSizeParser):
    """Class for parsing a packed value using struct.unpack"""
    
    def __init__(self, size: extendedReference, format_string: str, id: str):
        """Pass a size, a format string for the struct.unpack method, and an id"""
        super().__init__(id, self.getReference(size)._record)
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

    def updateRecord(self, value=None):
        if value is not None:
            try:
                packed = struct.pack(self.format_string, value)
                assert len(packed) == self.size
            except:
                raise ValueError(f'could not pack value {value} into size {self.size} for parser {self.id}')
        self._updateParent()

class IntegerParser(StructValueParser):
    """Class for parsing packed integers with a given size"""
    mapping = {
        1: 'b',
        2: 'h',
        4: 'i'
    }

    def __init__(self, id: str, size: extendedReference, little_endian: bool = True, signed=False):
        """Pass a valid size (number of bytes), endianness, and whether or not the value is signed or unsigned"""
        if not self.getReference(size)._record in self.mapping.keys():
            raise ValueError('Size not supported')
        
        format_string = self.mapping[size]
        if not signed:
            format_string = format_string.upper()

        if little_endian:
            format_string = '<' + format_string
        else:
            format_string = '>' + format_string
        
        super().__init__(size, format_string, id)

    def getAllRecords(self):
        return self._record

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
    """Subclass for parsing a variable number of bytes given by a reference"""

    def __init__(self, id: str, size: extendedReference):
        super().__init__(id)
        self.size = size
        self._size = None

    def getSize(self):
        """Retrieve a size from the record for already-parsed input"""
        if self.size is None:
            if self._record is None:
                return None
            else:
                return len(self._record)
        elif self._parent is None:
            return self.getReference(self.size)._record
        return self._parent.getReference(self.size)._record
    
    def __str__(self):
        return str(self._record)
    
    def getAllRecords(self):
        return str(self._record)
    
    def parse(self, buffer: BytesIO):
        self._record = buffer.read(self.getSize())

    def unparse(self, buffer: BytesIO):
        buffer.write(self._record)
    
    def updateRecord(self, value=None):
        """update the value stored in this record as well as the corresponding size record"""
        if value is not None:
            self._record = value
        if self._parent is not None:
            if self.size is not None:
                new_size = self._getInternalSize()
                size_parser = self._parent.getReference(self.size)
                size_parser._record = new_size
            self._parent.updateRecord()
        
class StringParser(ReferenceSizeParser):
    """Class for parsing reference-sized strings"""

    def parse(self, buffer: BytesIO):
        value = buffer.read(self.getSize())
        self._record = value.decode('utf-8')
    
    def unparse(self, buffer: BytesIO):
        buffer.write(self._record.encode('utf-8'))

    def _getInternalSize(self):
        return len(self._record.encode('utf-8'))
    
    def __str__(self):
        return str(self._record)
    
    def getAllRecords(self):
        return str(self._record)

class ImageParser(ReferenceSizeParser):
    """Class for parsing reference-sized image data"""

    def __init__(self, width: extendedReference, height: extendedReference, id: str, depth: extendedReference = 1, numColors: extendedReference = 4):
        """Create an image parser where depth is given in bytes and numColors is an integer referring to RGB vs RGBA, etc."""
        super().__init__(id, None)
        self.width = width
        self.height = height
        self.depth = depth
        self.numColors = numColors

    def _getDims(self):
        width = self._parent.getReference(self.width)._record
        height = self._parent.getReference(self.height)._record
        depth = self._parent.getReference(self.depth)._record
        return (width, height, depth)
    
    def _getNumColors(self):
        return self._parent.getReference(self.numColors)._record

    def getSize(self):
        width, height, depth = self._getDims()
        numColors = self._getNumColors()
        return width*height*depth*numColors
    
    def __str__(self):
        return str(self._record)
    
    def getAllRecords(self):
        return str(self._record)

    def _getMode(self):
        """Get Pillow mode"""
        _, _, depth = self._getDims()
        if depth != 1:
            raise ValueError('Currently only a byte-depth of 1 is supported')
        colorsToMode = {
            1: 'L',
            3: 'RGB',
            4: 'RGBA'
        }
        numColors = self._getNumColors()
        try:
            return colorsToMode[numColors]
        except KeyError:
            raise ValueError(f'Unsupported number of colors {numColors}')

    def parse(self, buffer: BytesIO):
        image_data = buffer.read(self.getSize())
        width, height, depth = self._getDims()
        mode = self._getMode()
        image = Image.frombytes(mode, (width, height), image_data)
        self._record = image

    def unparse(self, buffer: BytesIO):
        image_bytes = self._record.tobytes()
        buffer.write(image_bytes)

    def updateRecord(self, value: Image):
        raise NotImplementedError('no updating yet for image records')

class BlockParser(ReferenceSizeParser):
    """Class for parsing a block of parsable elements"""

    def __init__(self, id: str, elements: List[Union[FixedSizeParser, ReferenceSizeParser]], size=None):
        super().__init__(id, size)
        self._record: Dict[str, Parser] = OrderedDict()
        for element in elements:
            self._record[element.getID()] = element
            element._parent = self

    def getSize(self):
        if self.size is None:
            total = 0
            for id, element in self._record.items():
                size = element.getSize()
                if size is None:
                    return None
                total += size
            return total
        else:
            return super().getSize()

    def getAllRecords(self):
        record = {id: parser.getAllRecords() for id, parser in self._record.items()}
        return record

    def _getInternalSize(self):
        total = 0
        for parser in self.values():
            total += parser._getInternalSize()
        return total
    
    def __contains__(self, key):
        return key in self._record
    
    def keys(self):
        return self._record.keys()
    
    def values(self):
        return self._record.values()
    
    def items(self):
        return self._record.items()
    
    def __iter__(self):
        return iter(self._record)

    def __str__(self):
        return json.dumps(self.getAllRecords())
    
    def __len__(self):
        return len(self._record)
    
    def __getitem__(self, keys):
        if keys is None:
            raise ValueError('expected at least one key')
        if isinstance(keys, tuple):
            keys = list(keys)
        return self.getReference(keys)._record
        
    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            key = list(key)
        self.getReference(key).updateRecord(value)
        try:
            self.getReference(key).updateRecord(value)
        except KeyError:
            raise KeyError(f'{self.getID()} has no record with key {key}. Maybe you meant to use BlockParser.appendParser()')

    def __delitem__(self, key):
        del self._record[key]
        self.updateRecord()

    def appendParser(self, parser: Parser):
        id = parser.getID()
        assert id not in self
        self._record[parser.getID()] = parser

    def parse(self, buffer: BytesIO):
        size = self.getSize()
        if size is None:
            for id, element in self.items():
                element.parse(buffer)
        else:
            data = buffer.read(size)
            tempbuf = BytesIO(data)
            for id, element in self.items():
                element.parse(tempbuf)
            remainder = tempbuf.read()
            if len(remainder) > 0:
                self.appendParser(ReferenceSizeParser('_remainder', None))
                self._record['_remainder'].parse(BytesIO(remainder))
        # TODO change to avoid copy (use difference refs?)
        #  goal is to get rid of the reference size chunk parser
    
    def unparse(self, buffer: BytesIO):
        for id, element in self.items():
            element.unparse(buffer)
    
class ReferenceCountParser(Parser):

    def __init__(self, id: str, count_id: Reference, element_factory: Callable[[int], Parser]):
        super().__init__(id)
        self._record: List[Parser] = []
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
    
    def getAllRecords(self):
        records = [parser.getAllRecords() for parser in self._record]
        return records
    
    def __str__(self):
        l = self.getAllRecords()
        return json.dumps(l)
    
    def __len__(self):
        return self._count
    
    def __iter__(self):
        return iter(self._record)
    
    def __getitem__(self, idx: int):
        if isinstance(idx, tuple):
            idx = list(idx)
        return self.getReference(idx)
    
    def __setitem__(self, ref, value):
        self.getReference(ref).updateRecord(value)  

    def __delitem__(self, idx):
        del self._record[idx]
        self._updateCount()

    def parse(self, buffer: BytesIO):
        self._count = self._parent.getReference(self.count_id)._record
        for i in range(0, self._count):
            self._record.append(self.element_factory(i))
            self._record[i]._parent = self
            self._record[i].parse(buffer)

    def unparse(self, buffer: BytesIO):
        for element in self._record:
            element.unparse(buffer)

    def getCount(self):
        return len(self._record)

    def _updateCount(self):
        count = self.getCount()
        count_parser = self._parent.getReference(self.count_id)
        count_parser._count = count
        self._updateParent()

    def append(self, new_value):
        new_parser = self.element_factory(len(self._record))
        new_parser._record = new_value
        self._record.append(new_parser)
        self._updateCount()

    def insert(self, idx, value):
        initial_count = self.getCount
        new_parser = self.element_factory(idx)
        new_parser.updateRecord(value)
        self._record.insert(idx, new_parser)
        for i in range(idx+1, initial_count+1):
            self._record[i].id = i
        self._updateCount()

    def mapReference(self, reference: Reference, get_record: bool = False):
        if get_record:
            return [p.getReference(reference)._record for p in self]
        else:
            return [p.getReference(reference) for p in self]
    
class TransformationParser(BlockParser):

    def __init__(self, id: str, size: extendedReference, transform, transformInverse, elements: List[Parser]):
        super().__init__(id, elements, size)
        self.transform = transform
        self.transformInverse = transformInverse

    def parse(self, buffer: BytesIO):
        size = self.getSize()
        data = buffer.read(size)
        transformedBuffer = BytesIO(self.transform(data))
        self.size = None #prevent BlockParser.parse() from reading too few bytes
        super().parse(transformedBuffer)
        self.size = size
    
    def unparse(self, buffer: BytesIO):
        tempbuf = BytesIO()
        super().unparse(tempbuf)
        tempbuf.seek(0)
        buffer.write(self.transformInverse(tempbuf.read()))

class BytesExpansionParser(TransformationParser):
    """Class for expanding bytes to bits. Every byte gets expanded to 8 bytes each containing either x00 or x01.
    This is space inefficient but means that existing parser objects can be used on 'bits' as well"""

    def _expand(self, data: bytes, bit_sizes: List[int]):
        result = b''
        bits = ''.join([format(byte, '08b') for byte in data])
        for bit_size in bit_sizes:
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
        assert len(bits) / 8 == self.getSize()
        result = b''
        for i in range(0, self.getSize()):
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
        if key is None:
            return False
        try:
            self._active = self.mapping[key]
            self._active._parent = self._parent
            return True
        except KeyError:
            raise KeyError(f'ReferenceMappedParser read an unexpected key: {key}')

    def getSize(self):
        if not self._getParser():
            return None
        return self._active.getSize()
    
    def getAllRecords(self):
        if not self._getParser():
            raise KeyError('ReferenceMappedParser read a None record for key value')
        self._active._record = self._record
        return self._active.getAllRecords()
    
    def __str__(self):
        if not self._getParser():
            raise KeyError('ReferenceMappedParser read a None record for key value')
        self._active._record = self._record
        return str(self._active)
    
    def parse(self, buffer: BytesIO):
        if not self._getParser():
            raise KeyError('ReferenceMappedParser read a None record for key value')
        self._active.parse(buffer)
        self._record = self._active._record

    def unparse(self, buffer: BytesIO):
        if not self._getParser():
            raise KeyError('ReferenceMappedParser read a None record for key value')
        self._active._record = self._record
        self._active.unparse(buffer)
    
class EOFParser(Parser):

    def getSize(self):
        return None

    def __init__(self):
        super().__init__('_eof')
        self._record = 'EOF'

    def parse(self, buffer: BytesIO):
        remainder = buffer.read()
        if len(remainder) != 0:
            raise ValueError(f'Expected eof but found {len(remainder)} bytes instead')
    
    def unparse(self, buffer: BytesIO):
        pass

    def getAllRecords(self):
        return self._record

    def __str__(self):
        return str(self._record)
    
class HexParser(ReferenceSizeParser):

    def __init__(self, id: str, size: extendedReference = None, little_endian=True):
        super().__init__(id, size)
        self.little_endian = little_endian

    def parse(self, buffer: BytesIO):
        content = buffer.read(self.getSize())
        hex_bytes = [format(byte, '02X') for byte in content]
        if self.little_endian:
            hex_bytes = reversed(hex_bytes)
        self._record = ''.join(hex_bytes)
        
    def unparse(self, buffer: BytesIO):
        hex_bytes = bytes.fromhex(self._record)
        if self.little_endian:
            hex_bytes = bytes(reversed(hex_bytes))
        buffer.write(hex_bytes)
