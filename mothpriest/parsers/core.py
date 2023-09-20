import struct
from collections import OrderedDict
import warnings
from typing import Union, List, Callable, Dict
from functools import partial
from io import BytesIO
from PIL import Image
import json
from ..references import *

VERBOSE = False

# def changingOP(method):
#     def wrapper(instance: Parser, *args, **kwargs):
#         value = method(instance, *args, **kwargs)
#         instance._changed = True
#         return value
#     return wrapper

class Parser():
    """Abstract base class for parser objects"""

    def __init__(self, id: str, size: extendedReference = None, position: extendedReference = None, position_reference: extendedReference = None):
        self.id = id

        self._size = None
        self._position = None
        self._position_reference = position_reference
        self._parent: Union[Parser, None] = None
        self._prior: Union[Parser, None] = None
        self._subsequent: Union[Parser, None] = None
        self._record = None

        if size is not None:
            self.setSize(size)
        if position is not None:
            self.setPosition(position)

    def getID(self):
        """Returns the id of this element"""
        return self.id

    def setSize(self, size):
        self._size = size

    def getPrior(self):
        if self._prior is not None:
            return self._prior
        elif self._parent is not None:
            return self._parent.getPrior()
        else:
            None

    def getSize(self):
        if self._record is not None and (isinstance(self._size, int) or self._size is None): # We have an actual record and a constant size
            return self._getRecordSize()
        elif isinstance(self._size, int): # We have a fixed value
            return self._size
        elif self._size is None: # No record yet and size unspecified (greedy)
            return None
        else: # We have a reference
            self._size = Reference.fromOther(self._size)
            if self._parent is None:
                raise ValueError('cannot retrieve reference from orphan parser')
            sizeParser = self._parent.getReference(self._size)
            value = sizeParser.getRecord()
            if value is None:
                return None # If the size parser has no record yet, do nothing
            sizeParser._record = self.getSize
            self._size = value
            return self._size

    def _getRecordSize(self):
        if self._record is None:
            return None
        return len(self._record)

    def _setSize(self, value):
        self._size = value

    def getRecord(self):
        if callable(self._record):
            return self._record()
        return self._record

    def setRecord(self, value):
        if callable(self._record):
            import pdb; pdb.set_trace()
            raise ValueError('Cannot manually set a record which is referenced elsewhere')
        self._record = value

    def getPosition(self):
        prior = self.getPrior()
        if prior is None:
            return 0
        if self._record is not None and (isinstance(self._position, int) or self._position is None): # We have an actual record and a constant position
            priorSize = self.getPrior().getSize()
            if priorSize is None:
                return None
            priorPosition = self.getPrior().getPosition()
            if priorPosition is None:
                return None
            return priorPosition + priorSize
        elif isinstance(self._position, int): # We have a fixed value but no record yet
            return self._position
        elif self._position is None: # No record yet and position unspecified (ignored)
            return None
        else: # We have a reference
            self._position = Reference.fromOther(self._position)
            if self._parent is None:
                raise ValueError('cannot retrieve reference from orphan parser')
            positionParser = self._parent.getReference(self._position)
            value = positionParser.getRecord()
            if value is None:
                return None # If the size parser has no record yet, do nothing
            positionParser._record = self.getPosition
            self._position = value
            return self._position

    def setPosition(self, value):
        self._position = value

    # def _clearCache(self):
    #     self._cache = {}

    # def getPosition(self):
    #     if self.position is None:
    #         raise ValueError

    # def _getRoot(self):
    #     if self._parent is None:
    #         return self
    #     else:
    #         return self._parent._getRoot()

    # def getSize(self):
    #     """Retrieve a size from the record for already-parsed input"""
    #     if self.size is None:
    #         if self._record is None:
    #             return None
    #         else:
    #             return len(self._record)
    #     elif self._parent is None:
    #         return self.getReference(self.size)._record
    #     return self._parent.getReference(self.size)._record
    
    # def _setRecord(self, value):
    #     if value != self._record:
    #         self._record = value
    #         self._clearCache()
    #         self._changed = True
    
    def parse(self, buffer: BytesIO):
        parserPosition = self.getPosition()
        if parserPosition is not None:
            position = buffer.tell()
            if parserPosition < position:
                import pdb; pdb.set_trace()
                raise ValueError(f'parsing has already exceeded the position of parser {self.getID()}. \
                                 Currently at ({position}) and parser is positioned at ({parserPosition})')
            elif parserPosition > position:
                buffer.read(parserPosition - position)

        self.setRecord(buffer.read(self.getSize()))

    # def unparse(self, buffer: BytesIO):
    #     buffer.write(self._record)

    def getAllRecords(self):
        return str(self.getRecord())

    def __str__(self):
        """get a string representation of the record stored in this parser"""
        return str(self.getRecord())
    
    def __call__(self, buffer: BytesIO):
        self.parse(buffer)

    def getReference(self, reference: Union[Reference, str, int, List[str]]):
        """allows the use of strings and lists of strings in place of the corresponding Reference subclasses
        Return value indicates if anything was changed or not"""
        ref = Reference.fromOther(reference)
        return ref.retrieveRecord(self)

    # def _getRecordSize(self):
    #     """get the size currently store in self._record, which may not be the same value as self.getSize()"""
    #     if 'size' not in self._cache:
    #         with BytesIO() as dummy_stream:
    #             self.unparse(dummy_stream)
    #             new_size = dummy_stream.tell()
    #             self._cache['size'] = new_size
    #             return new_size
    #     else:
    #         return self._cache['size']

class MagicParser(Parser):
    """Class for parsing magic strings, which usually exist at the beginning of a file to hint at its file type"""

    def __init__(self, value: Union[str, bytes], id: str):
        """pass a size, the expected magic value, and an id"""
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
    
    def getAllRecords(self):
        return str(self.getRecord())

class StructValueParser(Parser):
    """Class for parsing a packed value using struct.unpack"""
    
    def __init__(self, size: extendedReference, format_string: str, id: str, position: extendedReference = None):
        """Pass a size, a format string for the struct.unpack method, and an id"""
        super().__init__(id, size, position=position)
        self.format_string = format_string

    def parse(self, buffer: BytesIO):
        """returns a list of unpacked values"""
        super().parse(buffer)
        self.setRecord(struct.unpack(self.format_string, self.getRecord()))
        # data = buffer.read(self.getSize())
        # self._record = struct.unpack(self.format_string, data)
    
    def _getRecordSize(self):
        return self._size

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

    def __init__(self, id: str, size: extendedReference, little_endian: bool = True, signed=False, position: extendedReference = None):
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
        
        super().__init__(size, format_string, id, position=position)

    def getAllRecords(self):
        return self.getRecord()

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
        
class StringParser(Parser):
    """Class for parsing reference-sized strings"""

    def parse(self, buffer: BytesIO):
        value = buffer.read(self.getSize())
        self._record = value.decode('utf-8')
    
    def unparse(self, buffer: BytesIO):
        buffer.write(self._record.encode('utf-8'))

    def _getRecordSize(self):
        return len(self._record.encode('utf-8'))
    
    def __str__(self):
        return str(self._record)
    
    def getAllRecords(self):
        return str(self.getRecord())

class ImageParser(Parser):
    """Class for parsing reference-sized image data"""

    def __init__(self, id: str, width: extendedReference, height: extendedReference, depth: extendedReference = 1, numColors: extendedReference = 4):
        """Create an image parser where depth is given in bytes and numColors is an integer referring to RGB vs RGBA, etc."""
        super().__init__(id)
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
        return str(self.getRecord())

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

class BlockParser(Parser):
    """Class for parsing a block of parsable elements"""

    def __init__(self, id: str, elements: List[Parser], size=None):
        super().__init__(id, size)
        self._record: Dict[str, Parser] = OrderedDict()
        prior = None
        for element in elements:
            element._prior = prior
            element._parent = self
            self._record[element.getID()] = element
            prior = element

    def getSize(self):
        if self._size is None:
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
        record = {id: parser.getAllRecords() for id, parser in self.getRecord().items()}
        return record

    def _getRecordSize(self):
        total = 0
        for parser in self.values():
            parser_size = parser._getRecordSize()
            if parser_size is None:
                return None
            total += parser_size
        return total

    def __contains__(self, key):
        return key in self.getRecord()

    def keys(self):
        return self.getRecord().keys()

    def values(self):
        return self.getRecord().values()

    def items(self):
        return self.getRecord().items()

    def __iter__(self):
        return iter(self.getRecord())

    def __str__(self):
        return json.dumps(self.getAllRecords())

    def __len__(self):
        return self.getSize()

    def __getitem__(self, keys):
        if keys is None:
            raise ValueError('expected at least one key')
        if isinstance(keys, tuple):
            keys = list(keys)
        return self.getReference(keys).getRecord()
        
    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            key = list(key)
        try:
            self.getReference(key).setRecord(value)
        except KeyError:
            raise KeyError(f'{self.getID()} has no record with key {key}. Maybe you meant to use BlockParser.appendParser()')

    def __delitem__(self, key):
        del self.getRecord()[key]

    def appendParser(self, parser: Parser):
        id = parser.getID()
        assert id not in self
        self.getRecord()[parser.getID()] = parser

    def parse(self, buffer: BytesIO):
        size = self.getSize()
        if size is None:
            for id, element in self.items():
                element.parse(buffer)
        else:
            data = buffer.read(size)
            with BytesIO(data) as tempbuf:
                for id, element in self.items():
                    element.parse(tempbuf)
                remainder = tempbuf.read()
                if len(remainder) > 0:
                    self.appendParser(Parser('_remainder', None))
                    with BytesIO(remainder) as remainder_buffer:
                        self._record['_remainder'].parse(remainder_buffer)
        # TODO change to avoid copy (use difference refs?)
        #  goal is to get rid of the reference size chunk parser

    def unparse(self, buffer: BytesIO):
        for id, element in self.items():
            element.unparse(buffer)

class ReferenceCountParser(Parser):

    def __init__(self, id: str, count_id: Reference, element_factory: Callable[[int], Parser], position: extendedReference = None):
        super().__init__(id, position=position)
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
        records = [parser.getAllRecords() for parser in self.getRecord()]
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
        self.getReference(ref).setRecord(value)

    def __delitem__(self, idx):
        del self._record[idx]

    def parse(self, buffer: BytesIO):
        self._count = self._parent.getReference(self.count_id)._record
        prior = None
        for i in range(0, self._count):
            self._record.append(self.element_factory(i))
            self._record[i]._parent = self
            self._record[i]._prior = prior
            prior = self._record[i]
            self._record[i].parse(buffer)

    def unparse(self, buffer: BytesIO):
        for element in self._record:
            element.unparse(buffer)

    def append(self, new_value):
        new_parser = self.element_factory(len(self._record))
        new_parser._record = new_value
        self._record.append(new_parser)

    def insert(self, idx, value):
        initial_count = self.getCount
        new_parser = self.element_factory(idx)
        new_parser.setRecord(value)
        self._record.insert(idx, new_parser)
        for i in range(idx+1, initial_count+1):
            self._record[i].id = i

    def mapReference(self, reference: Reference, get_record: bool = False):
        if get_record:
            return [p.getReference(reference)._record for p in self]
        else:
            return [p.getReference(reference) for p in self]
    
class TransformationParser(BlockParser):

    def __init__(self, id: str, size: extendedReference, transform, transformInverse, elements: List[Parser], in_place=False):
        super().__init__(id, elements, size)
        self.transform = transform
        self.transformInverse = transformInverse
        self.in_place = in_place

    def parse(self, buffer: BytesIO):
        size = self.getSize()
        start_position = buffer.tell()
        data = buffer.read(size)
        if not self.in_place:
            with BytesIO(self.transform(data)) as transformed_buffer:
                self._size = None #prevent BlockParser.parse() from reading too few bytes
                prior_fn = self.getPrior # prevent positioning issues
                self.getPrior = lambda: None
                super().parse(transformed_buffer)
                self._size = size
                self.getPrior = prior_fn
        else: # in place
            remainder = buffer.read()
            buffer.seek(start_position)
            buffer.write(self.transform(data))
            buffer.write(remainder)
            buffer.seek(start_position)
            self.size = None #prevent BlockParser.parse() from reading too few bytes
            super().parse(buffer)
            self.size = size

    def unparse(self, buffer: BytesIO):
        with BytesIO() as tempbuf:
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

    def _getRecordSize(self):
        return self._size

    def __init__(self, id: str, size: int, bit_sizes: List[int], parser: Parser):
        """Note that size here indicates the size in bytes, not bits"""
        if size != sum(bit_sizes) / 8:
            raise ValueError('Number of bits must add up to 1/8th the number of bytes')
        super().__init__(id, ConstIntegerReference(size), partial(self._expand, bit_sizes=bit_sizes), partial(self._contract, bit_sizes=bit_sizes), [parser])

class PDBParser(Parser):

    def __init__(self):
        super().__init__('pdb')
    
    def parse(self, buffer: BytesIO):
        import pdb; pdb.set_trace()
        pass

    def unparse(self, buffer: BytesIO):
        import pdb; pdb.set_trace()
        pass
    
class ReferenceMappedParser(Parser):

    def __init__(self, id: str, key_id: Reference, mapping: dict):
        super().__init__(id)
        self.key_id = key_id
        self.mapping = mapping
        self._active: Parser = None
        self._key = None

    def getKey(self):
        if self._active is None:
            keyParser = self._parent.getReference(self.key_id)
            if keyParser.getRecord() is None:
                return None
            self._key = keyParser.getRecord()
            keyParser._record = self.getKey
        return self._key

    def _getParser(self):
        key = self.getKey()
        if key is None:
            return False
        try:
            self._active = self.mapping[key]
            self._active._parent = self._parent
            return True
        except KeyError:
            raise KeyError(f'ReferenceMappedParser read an unexpected key: {key}')

    def switchActive(self, key):
        if key not in self.mapping:
            raise KeyError(f'key {key} not in mapping')
        self._getParser()
        self._record = self._active._record

    def getSize(self):
        if not self._getParser():
            return None
        return self._active.getSize()
    
    def getAllRecords(self):
        if not self._getParser():
            raise KeyError('ReferenceMappedParser read a None record for key value')
        self._active.setRecord(self.getRecord())
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
        # self._active._record = self._record
        self._active.unparse(buffer)

    def getReference(self, reference: extendedReference):
        if not self._getParser():
            raise KeyError('ReferenceMappedParser read a None record for key value')
        self._active._record = self._record
        return self._active.getReference(reference)
    
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
        # TODO find a way to enforce
        pass

    def getAllRecords(self):
        return self.getRecord()

    def __str__(self):
        return str(self._record)
    
class HexParser(Parser):

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

class NoneParser(Parser):
    """A parser which does absolutely nothing"""
    def __init__(self, id: str):
        super().__init__(id, 0)

    def parse(self, buffer: BytesIO):
        return
    
    def unparse(self, buffer: BytesIO):
        return

class ErrorParser(Parser):
    """A parser which raises an error on parse or unparse"""

    def __init__(self, id: str, error: Exception):
        super().__init__(id)
        self.error = error

    def parse(self, buffer: BytesIO):
        raise self.error

    def unparse(self, buffer: BytesIO):
        raise self.error

    def getSize(self):
        return 0
    
class SourceDeletingParser(BlockParser):
    """A parser which removes its source from the buffer after parsing. Otherwise it behaves like a BlockParser"""

    def parse(self, buffer:BytesIO):
        start = buffer.tell()
        super().parse(buffer)
        remainder = buffer.read()
        buffer.seek(start)
        buffer.write(remainder)
        buffer.truncate()
        buffer.seek(start)