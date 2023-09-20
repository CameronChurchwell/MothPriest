from mothpriest.parsers import Parser, BlockParser, IntegerParser, EOFParser, PDBParser
from io import BytesIO

def test_basic():
    data = b'\x00\x01\x02'
    id = 'basic'
    parser = Parser(id)
    with BytesIO(data) as buffer:
        parser(buffer)
    
    assert parser._record == data
    assert parser.getAllRecords() == str(data)
    assert str(parser) == str(data)
    assert parser.getID() == id
    assert parser.getSize() == len(data)
    # parser.__updatePositions()
    # assert parser._position == 0

def test_new():
    data = b'\x00\x03\xAE\xAA\xA1\x00\xBB'
    parser = BlockParser(
        'base',
        [
            IntegerParser('size', size=2, little_endian=False),
            Parser('content', 'size'),
            Parser('last', position=6),
            EOFParser()
        ]
    )

    with BytesIO(data) as buffer:
        parser(buffer)

    assert parser['size'] == 3
    assert parser['content'] == b'\xAE\xAA\xA1'
    assert parser['last'] == b'\xBB'
    assert parser.getReference('last').getPosition() == 5

    parser['content'] = b'\xAE\xAA\xA1\xAA'

    assert parser['size'] == 4
    assert parser['content'] == b'\xAE\xAA\xA1\xAA'
    assert parser['last'] == b'\xBB'
    assert parser.getReference('last').getPosition() == 6
