import struct
from collections import OrderedDict
from typing import Union, List, Callable, Dict
from functools import partial
from io import BytesIO
from PIL import Image
import json
import hashlib

from mothpriest.references import List
from ..references import *
from ..buffers import DivergentBuffer

VERBOSE = False

class Parser():
    """Abstract base class for parser objects"""

    def __init__(self, id: str, size: extendedReference = None, position: extendedReference = None):
        self.id = id

        self._size = size
        self.sizeParser = None

        self._position = position
        self.positionParser = None

        self._parent: Union[Parser, None] = None
        self._record = None

    def getID(self):
        """Returns the id of this parser"""
        return self.id

    def getSize(self):
        """Calculates the best guess for the current size of this parser in bytes. See documentation on References for more information"""
        if self._record is not None and (isinstance(self._size, int) or self._size is None): # We have an actual record and a constant size
            return self._size
        if isinstance(self._size, int): # We have a fixed value
            return self._size
        elif self._size is None: # No record yet and size unspecified (greedy)
            return None
        else: # We have a reference
            # Convert from extendedReference to Reference
            self._size = Reference.fromOther(self._size)
            # Assure self is not an orphan
            if self._parent is None:
                raise ValueError('cannot retrieve reference from orphan parser')
            # Get the referenced parser for the size
            sizeParser = self._parent.getReference(self._size)
            # Get the value stored in the reference
            value = sizeParser.getRecord()
            # If the size parser has no record yet, do nothing
            if value is None:
                return None
            # Link the size parser to this object
            sizeParser._record = self.getSize
            if hasattr(sizeParser, 'deferUnparsing'):
                sizeParser.deferUnparsing()
                self.sizeParser = sizeParser
            # Now store the actual size in this object
            self._size = value
            # Return that size
            return self._size

    def getRecord(self):
        """Returns the value of the record of this parser. 
        If self._record stores a function, it returns the return value fo that function"""
        if callable(self._record):
            return self._record()
        return self._record

    def setRecord(self, value):
        """Set the record stored in this parser"""
        if callable(self._record):
            raise ValueError('Cannot manually set a record which is referenced elsewhere')
        self._record = value

    def getPosition(self):
        """Calculates the best guess for the current size of this parser in bytes. See documentation on References for more information"""
        if self._record is not None and (isinstance(self._position, int) or self._position is None): # We have an actual record and a constant position
            return self._position
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
            if hasattr(positionParser, 'deferUnparsing'):
                positionParser.deferUnparsing()
                self.positionParser = positionParser
            self._position = value
            return self._position

    def parse(self, buffer: BytesIO):
        """Parse a buffer of bytes. The buffer must be readable, seekable, writable, and allow backwards seeking"""
        parserPosition = self.getPosition()
        if parserPosition is not None:
            position = buffer.tell()
            if parserPosition < position:
                raise ValueError(f'parsing has already exceeded the position of parser {self.getID()}. \
                                 Currently at ({position}) and parser is positioned at ({parserPosition})')
            elif parserPosition > position:
                buffer.read(parserPosition - position)

        self.setRecord(buffer.read(self.getSize()))

    def unparse(self, buffer: BytesIO):
        """Write the record stored in this parser out to a buffer. This should reverse the actions of `parse()`"""
        startPosition = buffer.tell()
        buffer.write(self.getRecord())
        endPosition = buffer.tell()
        self._size = endPosition-startPosition
        self._position = startPosition
        if self.sizeParser is not None:
            # call the backtrackingUnparse method wrapping sizeParser.unparse
            self.sizeParser.unparse(buffer)
            # re-defer unparsing on the sizeParser
            self.sizeParser.deferUnparsing()
        if self.positionParser is not None:
            # call the backtrackingUnparse method wrapping positionParser.unparse
            self.positionParser.unparse(buffer)
            # re-defer unparsing on the positionParser
            self.positionParser.deferUnparsing()

    def deferUnparsing(self):
        """Defer unparsing until later"""

        originalUnparse = self.unparse

        def deferredUnparse(buffer: BytesIO):
            startPosition = buffer.tell()

            # We always want to make sure we write to the same buffer when we backtrack
            deferredBuffer = buffer

            # Write temporary blank bytes
            buffer.write(b"\x00"*self.getSize())

            def backtrackingUnparse(buffer: BytesIO):
                returnPosition = deferredBuffer.tell()

                deferredBuffer.seek(startPosition)
                originalUnparse(deferredBuffer)
                deferredBuffer.seek(returnPosition)

                self.unparse = originalUnparse

            self.unparse = backtrackingUnparse

        self.unparse = deferredUnparse

    def getAllRecords(self):
        """Return all of the records stored in a parser and its children"""
        return str(self.getRecord())

    def __str__(self):
        """get a string representation of the record stored in this parser"""
        return str(self.getRecord())

    def __call__(self, buffer: BytesIO):
        """Equivalent to `Parser.parse`"""
        self.parse(buffer)

    def getReference(self, reference: extendedReference):
        """Retrieve a record value from a reference, starting the search from this parser"""
        ref = Reference.fromOther(reference)
        return ref.retrieveRecord(self)

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
        if self.getRecord() is None or len(self.getRecord()) == 0:
            self.getRecord()
        self.setRecord(struct.unpack(self.format_string, self.getRecord()))

    def unparse(self, buffer: BytesIO):
        try:
            packed = struct.pack(self.format_string, *self.getRecord())
        except TypeError:
            packed = struct.pack(self.format_string, self.getRecord())
            self.getRecord()
        startPosition = buffer.tell()
        buffer.write(packed)
        endPosition = buffer.tell()
        self._size = endPosition-startPosition
        self._position = startPosition
        if self.sizeParser is not None:
            # call the backtrackingUnparse method wrapping sizeParser.unparse
            self.sizeParser.unparse(buffer)
            # re-defer unparsing on the sizeParser
            self.sizeParser.deferUnparsing()
        if self.positionParser is not None:
            # call the backtrackingUnparse method wrapping positionParser.unparse
            self.positionParser.unparse(buffer)
            # re-defer unparsing on the positionParser
            self.positionParser.deferUnparsing()

class IntegerParser(Parser):
    """Class for parsing packed integers with a given size"""

    def __init__(self, id: str, size: extendedReference, little_endian: bool = True, signed=False, position: extendedReference = None):
        """Pass a valid size (number of bytes), endianness, and whether or not the value is signed or unsigned"""
        super().__init__(id, size=size, position=position)
        self.little_endian = little_endian
        self.signed = signed

    def getAllRecords(self):
        return self.getRecord()

    def parse(self, buffer: BytesIO):
        """parses a single integer"""
        super().parse(buffer)
        self._record = int.from_bytes(
            self._record, 
            byteorder = 'little' if self.little_endian else 'big',
            signed = self.signed
        )

    def unparse(self, buffer: BytesIO):
        record = self._record
        self._record = self.getRecord().to_bytes(
            self.getSize(),
            byteorder = 'little' if self.little_endian else 'big',
            signed = self.signed
        )
        super().unparse(buffer)
        self._record = record


class StringParser(Parser):
    """Class for parsing reference-sized strings"""

    def parse(self, buffer: BytesIO):
        value = buffer.read(self.getSize())
        self._record = value.decode('utf-8')

    def unparse(self, buffer: BytesIO):
        startPosition = buffer.tell()
        buffer.write(self.getRecord().encode('utf-8'))
        endPosition = buffer.tell()
        self._size = endPosition-startPosition
        self._position = startPosition
        if self.sizeParser is not None:
            # call the backtrackingUnparse method wrapping sizeParser.unparse
            self.sizeParser.unparse(buffer)
            # re-defer unparsing on the sizeParser
            self.sizeParser.deferUnparsing()
        if self.positionParser is not None:
            # call the backtrackingUnparse method wrapping positionParser.unparse
            self.positionParser.unparse(buffer)
            # re-defer unparsing on the positionParser
            self.positionParser.deferUnparsing()

    def __str__(self):
        return str(self.getRecord())

    def getAllRecords(self):
        return str(self.getRecord())

class ImageParser(Parser):
    """Class for parsing reference-sized image data"""
    #TODO update references to do deferred unparsing

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
        image_bytes = self.getRecord().tobytes()
        #TODO this ignores size and position changes and deferred parsing?
        buffer.write(image_bytes)

class BlockParser(Parser):
    """Class for parsing a block of parsable elements"""

    def __init__(self, id: str, elements: List[Parser], size=None, position=None):
        super().__init__(id, size, position)
        self._record: Dict[str, Parser] = OrderedDict()
        for element in elements:
            element._parent = self
            self._record[element.getID()] = element

    def getSize(self):
        if self._size is None: #TODO change this to enforce correct size?
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
        """Append a parser to the end of the block"""
        id = parser.getID()
        assert id not in self
        self.getRecord()[parser.getID()] = parser

    def parse(self, buffer: BytesIO):
        size = self.getSize()
        for id, element in self.items():
            element.parse(buffer)

    def unparse(self, buffer: BytesIO):
        startPosition = buffer.tell()
        for id, element in self.items():
            element.unparse(buffer)
        endPosition = buffer.tell()
        self._size = endPosition-startPosition
        self._position = startPosition
        if self.sizeParser is not None:
            # call the backtrackingUnparse method wrapping sizeParser.unparse
            self.sizeParser.unparse(buffer)
            # re-defer unparsing on the sizeParser
            self.sizeParser.deferUnparsing()
        if self.positionParser is not None:
            # call the backtrackingUnparse method wrapping positionParser.unparse
            self.positionParser.unparse(buffer)
            # re-defer unparsing on the positionParser
            self.positionParser.deferUnparsing()

class ReferenceCountParser(Parser):
    """Parser for parsing a reference-determined amount of the same structure/entry"""

    def __init__(self, id: str, count_id: Reference, element_factory: Callable[[int], Parser], position: extendedReference = None):
        super().__init__(id, position=position)
        self._record: List[Parser] = []
        self.count_id = count_id
        self.element_factory = element_factory
        self._count = -1
        self.countParser = None

    def getSize(self):
        count = self.getCount()
        if count is None:
            return None
        if len(self.getRecord()) != count:
            raise ValueError(f'length of self._record ({len(self.getRecord())}) must be equal to self._count ({self.getCount()})')
        total = 0
        for element in self.getRecord():
            size = element.getSize()
            total += size
        return total

    def _getRecordSize(self):
        total = 0
        for element in self.getRecord():
            size = element._getRecordSize()
            if size is None:
                return None
            total += size
        return total

    def getCount(self):
        if self._count == -1: # We don't have a count yet, get from reference
            countParser = self._parent.getReference(self.count_id)
            value = countParser.getRecord()
            if value is None:
                return None
            countParser._record = self.getCount
            if hasattr(countParser, 'deferUnparsing'):
                self.countParser = countParser
                self.countParser.deferUnparsing()
            self._count = value
            return self._count
        self._count = len(self.getRecord())
        return self._count

    def getAllRecords(self):
        records = [parser.getAllRecords() for parser in self.getRecord()]
        return records

    def __str__(self):
        l = self.getAllRecords()
        return json.dumps(l)

    def __len__(self):
        return self.getCount() #TODO potentially really confusing

    def __iter__(self):
        return iter(self._record)

    def __getitem__(self, idx: int):
        if isinstance(idx, tuple):
            idx = list(idx)
        return self.getReference(idx)

    def __setitem__(self, ref, value):
        self.getReference(ref).setRecord(value)

    def __delitem__(self, idx):
        del self.getRecord()[idx]

    def parse(self, buffer: BytesIO):
        parserPosition = self.getPosition()
        if parserPosition is not None:
            position = buffer.tell()
            if parserPosition < position:
                raise ValueError(f'parsing has already exceeded the position of parser {self.getID()}. \
                                 Currently at ({position}) and parser is positioned at ({parserPosition})')
            elif parserPosition > position:
                buffer.read(parserPosition - position)
        count = self.getCount()
        record = self.getRecord()
        for i in range(0, count):
            record.append(self.element_factory(i))
            record[i]._parent = self
            record[i].parse(buffer)

    def unparse(self, buffer: BytesIO):
        elements = self.getRecord()
        self._count = len(elements)
        # iterator = tqdm.tqdm(elements, desc=self.getID(), total=len(elements))
        startPosition = buffer.tell()
        for element in elements:
            element.unparse(buffer)
        endPosition = buffer.tell()
        self._size = endPosition-startPosition
        self._position = startPosition
        if self.positionParser is not None:
            # call the backtrackingUnparse method wrapping positionParser.unparse
            self.positionParser.unparse(buffer)
            # re-defer unparsing on the positionParser
            self.positionParser.deferUnparsing()
        if self.countParser is not None:
            # call the backtrackingUnparse method wrapping countParser.unparse
            self.countParser.unparse(buffer)
            # re-defer unparsing ont the countParser
            self.countParser.deferUnparsing()

    def append(self, new_value):
        """Append a new parser to the end of the block and set its record to `new_value`"""
        new_parser = self.element_factory(self.getCount)
        new_parser.setRecord(new_value)
        self.getRecord().append(new_parser)

    def insert(self, idx, new_value):
        """Insert a new parser into the block and set its record to `new_value`"""
        initial_count = self.getCount()
        new_parser = self.element_factory(idx)
        new_parser.setRecord(new_value)
        self.getRecord().insert(idx, new_parser)
        for i in range(idx+1, initial_count+1):
            self.getRecord()[i].id = i

    def mapReference(self, reference: Reference, get_record: bool = False):
        """Get the value of a reference when mapped onto every child of this parser"""
        if get_record:
            return [p.getReference(reference)._record for p in self]
        else:
            return [p.getReference(reference) for p in self]

class TransformationParser(BlockParser):
    """Apply a transformation to the bytes of a buffer before parsing. Similarly, applies an inverse transform after unparsing.
    Otherwise, this behaves like a BlockParser"""

    def __init__(self, id: str, size: extendedReference, transform, transformInverse, elements: List[Parser], position: extendedReference=None, in_place=False):
        super().__init__(id, elements, size, position)
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
                super().parse(transformed_buffer)
                self._size = size
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
        offset = buffer.tell()
        with DivergentBuffer(buffer, offset=offset) as temp_buf:
            # prevent sizeParser from being written with transformed size
            sizeParser = self.sizeParser
            self.sizeParser = None

            # prevent size from being overwritten with transformed size
            size = self._size

            super().unparse(temp_buf)

            # restore size
            self._size = size

            # restore sizeParser
            self.sizeParser = sizeParser

            # go to offset in temporary buffer
            temp_buf.seek(offset)

            # copy into actual buffer
            buffer.seek(offset)
            startPosition = offset
            buffer.write(self.transformInverse(temp_buf.read()))
            endPosition = buffer.tell()
        self._size = endPosition-startPosition
        if self.sizeParser is not None:
            # call the backtrackingUnparse method wrapping sizeParser.unparse
            self.sizeParser.unparse(buffer)
            # re-defer unparsing on the sizeParser
            self.sizeParser.deferUnparsing()

class BytesExpansionParser(TransformationParser):
    """Class for expanding bytes to bits. Every byte gets expanded to 8 bytes each containing either x00 or x01.
    This is space inefficient but means that existing parser objects can be used on 'bits' as well""" #TODO update

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
        assert len(bits) / 8 == self.getSize(), breakpoint()
        result = b''
        for i in range(0, self.getSize()):
            result += struct.pack('B', int(bits[i*8:i*8+8], 2))
        return result

    def __init__(self, id: str, size: int, bit_sizes: List[int], parser: Parser):
        """Note that size here indicates the size in bytes, not bits"""
        if size != sum(bit_sizes) / 8:
            raise ValueError('Number of bits must add up to 1/8th the number of bytes')
        super().__init__(id, ConstIntegerReference(size), partial(self._expand, bit_sizes=bit_sizes), partial(self._contract, bit_sizes=bit_sizes), [parser])

class PDBParser(Parser):
    """A parser which sets a breakpoint in parsing and unparsing"""

    def __init__(self):
        super().__init__('pdb')
        self._record = 'pdb'

    def parse(self, buffer: BytesIO):
        import pdb; pdb.set_trace()
        pass

    def unparse(self, buffer: BytesIO):
        import pdb; pdb.set_trace()
        pass

    def getSize(self):
        return 0

class ReferenceMappedParser(Parser):
    """A parser which uses a reference and dictionary of parsers to determine which dictionary entry to use to parse"""

    def __init__(self, id: str, key_id: Reference, mapping: dict, transfer_record=True):
        super().__init__(id)
        self.key_id = key_id
        self.mapping = mapping
        self._active: Parser = None
        self._key = None
        self._transfer_record = transfer_record

    def getKey(self):
        """Get the value of the key determining the current active parser"""
        if self._active is None:
            keyParser = self._parent.getReference(self.key_id)
            if keyParser.getRecord() is None:
                return None
            self._key = keyParser.getRecord()
            keyParser._record = self.getKey
        return self._key

    def setKey(self, key):
        """Set the value of the key determining the current active parser"""
        if key not in self.mapping:
            raise KeyError(f'key {key} not in mapping')
        self._key = key
        self._getParser()

    def _getParser(self):
        """Ensure that the currently active parser is the one pointed to by the current key"""
        key = self.getKey()
        if key is None:
            return False
        try:
            if self._active is not None:
                self._record = self._active._record
            self._active = self.mapping[key]
            self._active._parent = self._parent
            if self._transfer_record:
                self._active._record = self._record
            else:
                self._record = self._active._record
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
        return self._active.getAllRecords()

    def __str__(self):
        if not self._getParser():
            raise KeyError('ReferenceMappedParser read a None record for key value')
        return str(self._active)

    def parse(self, buffer: BytesIO):
        if not self._getParser():
            raise KeyError('ReferenceMappedParser read a None record for key value')
        self._active.parse(buffer)
        self.setRecord(self._active.getRecord())

    def unparse(self, buffer: BytesIO):
        if not self._getParser():
            raise KeyError('ReferenceMappedParser read a None record for key value')
        self._active.unparse(buffer)

    def getReference(self, reference: extendedReference):
        if not self._getParser():
            raise KeyError('ReferenceMappedParser read a None record for key value')
        return self._active.getReference(reference)

class EOFParser(Parser):
    """Parser which enforces EOF during parsing, but currently not during unparsing"""

    def getSize(self):
        return None # greedy, always read everything remaining

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
    """A parser for reading hex values"""

    def __init__(self, id: str, size: extendedReference = None, position: extendedReference = None, little_endian=True):
        super().__init__(id, size)
        self.little_endian = little_endian

    def parse(self, buffer: BytesIO):
        content = buffer.read(self.getSize())
        hex_bytes = [format(byte, '02X') for byte in content]
        if self.little_endian:
            hex_bytes = reversed(hex_bytes)
        self.setRecord(''.join(hex_bytes))

    def unparse(self, buffer: BytesIO):
        hex_bytes = bytes.fromhex(self.getRecord())
        if self.little_endian:
            hex_bytes = bytes(reversed(hex_bytes))
        rawRecord = self.getRecord()
        self._record = hex_bytes
        super().unparse(buffer)
        self._record = rawRecord

class NoneParser(Parser):
    """A parser which does absolutely nothing"""
    def __init__(self, id: str):
        super().__init__(id, size=0)

    def parse(self, buffer: BytesIO):
        return

    def unparse(self, buffer: BytesIO):
        return

class ErrorParser(Parser):
    """A parser which raises an error on parse or unparse. Useful in conjunction with `ReferenceMappedParser`"""

    def __init__(self, id: str, error: Exception):
        super().__init__(id)
        self.error = error

    def parse(self, buffer: BytesIO):
        raise self.error

    def unparse(self, buffer: BytesIO):
        raise self.error

    def getSize(self):
        return 0

class BackFoldingParser(BlockParser):
    """A parser which folds the buffer backward over itself a given number of bytes. This is helpful if you have,
    for example, compression sizes which are not factored into the offsets of tables once the files is uncompressed.
    """

    def __init__(self, id: str, elements: List[Parser], size=None, position=None, foldSize: int = 0):
        super().__init__(id, elements, size, position)
        self.foldSize = foldSize
        assert foldSize >= 0 and isinstance(foldSize, int)

    def parse(self, buffer: BytesIO):
        currentPosition = buffer.tell()
        regionStart = currentPosition - self.foldSize
        regionEnd = currentPosition
        with BytesIO() as tempBuf:
            # copy buffer up to region start
            buffer.seek(0)
            tempBuf.write(buffer.read(regionStart))
            assert tempBuf.tell() == regionStart

            # skip fold region
            buffer.seek(regionEnd)

            # copy remainder of buffer
            tempBuf.write(buffer.read())

            # seek back to fold (regionStart to regionEnd collapses to regionStart)
            tempBuf.seek(regionStart)

            # parse using folded buffer
            super().parse(tempBuf)

            # seek to sync position of buffer to that of tempBuf, but offset by the foldSize
            buffer.seek(tempBuf.tell() + self.foldSize)

    def unparse(self, buffer: BytesIO):
        currentPosition = buffer.tell()
        regionStart = currentPosition - self.foldSize
        regionEnd = currentPosition
        with BytesIO() as tempBuf:
            # copy buffer up to region start
            buffer.seek(0)
            tempBuf.write(buffer.read(regionStart))

            # unparse into folded buffer (folded region not present)
            super().unparse(tempBuf)

            # seek back to fold in folded buffer
            tempBuf.seek(regionStart)

            # seek to end of fold in original buffer (skip over the fold so it is not overwritten)
            buffer.seek(regionEnd)

            # copy unparsed content back to original buffer
            buffer.write(tempBuf.read())

class IntegrityParser(BlockParser):
    """A parser for handling checksums that are built into files. Functions like a BlockParser"""

    def __init__(self, id: str, elements: List[Parser], checksumFunction: Union[Callable, str], checksum: extendedReference, size=None, position=None):
        """checksumFunction can be either a callable or the name of a hash function accessible in the top level of hashlib (e.g. 'md5')"""
        super().__init__(id, elements, size, position)
        if callable(checksumFunction):
            self.checksumFunction = checksumFunction
        elif isinstance(checksumFunction, str):
            try:
                hashFunction = getattr(hashlib, checksumFunction)
                def checksumInt(x: bytes):
                    return int(hashFunction(x).hexdigest(), 16)
                self.checksumFunction = checksumInt
            except AttributeError:
                raise AttributeError(f"Could not find hash function {checksumFunction} in hashlib, try supplying manually")
        else:
            raise ValueError("checksumFunction must be either a function or a string")
        self._checksum = checksum
        self.checksumParser = None

    def getChecksum(self):
        """Calculates the best guess for the current checksum of this parser. See documentation on References for more information"""
        if isinstance(self._checksum, int):
            return self._checksum
        else:
            self._checksum = Reference.fromOther(self._checksum)
            # Assure self is not an orphan
            if self._parent is None:
                raise ValueError('cannot retrieve reference from orphan parser')
            # Get the referenced parser for the checksum
            checksumParser = self._parent.getReference(self._checksum)
            # Get the value stored in the reference
            value = checksumParser.getRecord()
            # If the checksum parser has no record yet, do nothing
            if value is None:
                raise ValueError('Checksum parser has no record, cannot get checksum value')
            # Link the size parser to this object
            checksumParser._record = self.getChecksum
            if hasattr(checksumParser, 'deferUnparsing'):
                checksumParser.deferUnparsing()
                self.checksumParser = checksumParser
            # Now store the actual size in this object
            self._checksum = value
            # Return that size
            return self._checksum

    def parse(self, buffer: BytesIO):
        startPosition = buffer.tell()
        super().parse(buffer)
        endPosition = buffer.tell()

        checksum = self.getChecksum()

        buffer.seek(startPosition)
        content = buffer.read(endPosition-startPosition)

        computed = self.checksumFunction(content)
        if computed != checksum:
            raise ValueError(f"Checksum validation failed for {self.getID()}: {computed} != {checksum}")

    def unparse(self, buffer: BytesIO):
        startPosition = buffer.tell()
        super().unparse(buffer)
        endPosition = buffer.tell()

        buffer.seek(startPosition)
        content = buffer.read(endPosition-startPosition)

        computed = self.checksumFunction(content)

        self._checksum = computed
        self.checksumParser.unparse(buffer)
        self.checksumParser.deferUnparsing()