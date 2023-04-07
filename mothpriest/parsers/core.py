from abc import ABC, abstractmethod
import struct
from typing import Union, List
from functools import partial
from tqdm import tqdm
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
    def parse(self, data: bytes):
        """Parse input bytes"""
        pass

    @abstractmethod
    def unparse(self):
        """Unparse record back into bytes"""
        pass

    @abstractmethod
    def getSize(self):
        """get size of this element"""
        pass

    # def getReference(self, record, reference):
    #     #TODO refactor to use recursion
    #     #TODO add support of lists, not just dictionaries
    #     """retrieve a referenced value given a record and a reference"""
    #     if record is None:
    #         raise ValueError('no record passed')
    #     if isinstance(reference, str):
    #         if reference not in record.keys():
    #             raise ValueError(f'failed to find reference {reference} for object {self.id}')
    #         return record[reference]
    #     else:
    #         try:
    #             for k in reference:
    #                 record = record[k]
    #             return record
    #         except KeyError:
    #             raise ValueError(f"failed to find referenced size {' -> '.join(reference)} for object {self.id}")

    # def getReference(self, record, reference):
    #     """retrieve a referenced value given a record and a reference"""
    #     #TODO refactor to use recursion
    #     #TODO add support of lists, not just dictionaries
    #     if record is None:
    #         raise ValueError(f'no record passed for element with id {self.id}')
    #     if isinstance(reference, list):
    #         if len(reference) == 1:
    #             return self.getReference(record, reference[0])
    #         else:
    #             subRecord = self.getReference(
    #                 record,
    #                 reference[0]
    #             )
    #             if subRecord is None:
    #                 raise ValueError('subRecord is None')
    #             return self.getReference(
    #                 subRecord,
    #                 reference[1:]
    #             )
    #     elif (isinstance(reference, int) and isinstance(record, list)) or (isinstance(reference, str) and isinstance(record, dict)):
    #         return record[reference]
    #     else:
    #         raise ValueError('Invalid (record, reference) pair')

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
    
    def unparse(self):
        return self._record[self.id]

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
    
    def unparse(self):
        return self.value

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
    
    def unparse(self):
        return struct.pack(self.unpack_string, self._record)
    
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
    
    def unparse(self):
        return self._record.encode('utf-8')
    
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
    
    def unparse(self):
        return self._record

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

    # #@profile
    # def parse(self, data):
    #     record = {'_parent': self._record}

    #     data_offset = 0
    #     for element in self.elements:
    #         element.updateRecord(record)
    #         #get initial size
    #         size = element.getSize()
    #         sub_data = data[data_offset:][:size]
    #         parsed = element.parse(sub_data)
    #         if element.id is not None:
    #             record[element.id] = parsed
    #         #update size
    #         element.updateRecord(record)
    #         size = element.getSize()
    #         data_offset += size
    #     del record['_parent']
    #     return record

    def parse(self, buffer: BytesIO):
        record = {'_parent': self._record}
        for element in self.elements:
            element.updateRecord(record)
            record[element.getId()] = element.parse(buffer)
        del record['_parent']
        return record
    
    def unparse(self, record: dict):
        result = b''
        for element in self.elements:
            if element.id is not None:
                result += element.unparse(record[element.id])
            else:
                result += element.unparse(record)
        return result

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

    def unparse(self, record):
        return record
    
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

    #@profile
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
        iterator = tqdm(
            self._range,
            total=self._count,
            desc=self.id,
            dynamic_ncols=True
        )
        ep = self.element_parser
        ep.updateRecord(record)
        try:
            for _ in iterator:
                ep.updateRecord(record)
                element_record = ep.parse(buffer)
                record.append(element_record)
        except TypeError:
            import pdb; pdb.set_trace()
        return record
        

    def unparse(self, record):
        id_base = self.element_parser.id
        for i in range(0, self._count):
            self.element_parser.id = id_base + f'_{i}'
            if self.element_parser.id is not None:
                result += self.element_parser.unparse(record[self.element_parser.id])
            else:
                result += self.element_parser.unparse(record)
        self.element_parser.id = id_base
        return result


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

    # def parse(self, data: bytes):
    #     result = {}
    #     result[self.parser.id] = self.parser.parse(self.transform(data))
    #     return result

    def parse(self, buffer: BytesIO):
        size = self.getSize()
        data = buffer.read(size)
        transformed = BytesIO(self.transform(data))
        return self.parser.parse(transformed)
    
    def unparse(self):
        return self.transformInverse(self.parser.unparse(self._record[self.parser.id]))
    
class FixedPaddingParser(FixedSizeParser):

    def __init__(self, size: int):
        super().__init__('padding', size)

    def parse(self, buffer: BytesIO):
        return buffer.read(self.getSize())

    def unparse(self, record):
        new_padding = b'\x00' * self.size
        # print(len(new_padding), new_padding)
        return new_padding

class BytesExpansionParser(FixedSizeParser, TransformationParser):
    """Class for expanding bytes to bits. Every byte gets expanded to 8 bytes each containing either x00 or x01.
    This is space inefficient but means that existing parser objects can be used on 'bits' as well"""

    #@profile
    def _expand(self, data: bytes, bit_sizes: List[int]):
        #TODO does not work if one set of bits spans two or more bytes?
        result = bytearray(len(bit_sizes))
        bit_size_index = 0
        result_index = 0
        for binary in [format(byte, '08b') for byte in data]:
            # binary = bin(byte)[2:].rjust(8, '0')
            binary_index = 0
            while binary_index < 8:
                bit_size = bit_sizes[bit_size_index]
                bit_size_index += 1
                bits = binary[binary_index:binary_index+bit_size]
                binary_index += bit_size
                value = int(bits, 2)
                # result += struct.pack('<' + 'B' * ((bit_size + 8) // 8), value)
                result[result_index] = value
                result_index += 1
        return bytes(result)


    def __init__(self, id: str, size: int, bit_sizes: List[int], parser: FixedSizeParser):
        """Note that size here indicates the size in bytes, not bits"""

        if size != sum(bit_sizes) / 8:
            raise ValueError('Number of bits must add up to 1/8th the number of bytes')

        self.id = id
        self.size = size
        self._record = {}
        self.transform = partial(self._expand, bit_sizes=bit_sizes)
        self.transformInverse = None
        self.parser = parser

    def getSize(self):
        return super().getSize()
    
class PDBParser(Parser):

    def getSize(self):
        return None
    
    def parse(self, buffer: BytesIO):
        import pdb; pdb.set_trace()
        pass

    def unparse(self, record):
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
    
    def unparse(self, record):
        return self._active.unparse(record)
    
class ReferenceSizeChunkParser(ReferenceSizeParser):
    """Class for chunking the input bytes to allow for chunks to only be partially read if their size is known ahead of time"""

    def __init__(self, id: str, size_id: Reference, parser: Parser):
        super().__init__(id, size_id)
        self.parser = parser

    def parse(self, buffer: BytesIO):
        chunk = BytesIO(buffer.read(self.getSize()))
        self.parser.updateRecord(self._record)
        return self.parser.parse(chunk)
    
    def unparse(self, record):
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
    
    def unparse(self):
        raise NotImplementedError('not yet implemented')
    
#TODO add macros to this package
