from mothpriest.parsers import Parser, BlockParser, IntegerParser, EOFParser, PDBParser, BackFoldingParser, StringParser
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

def test_fold():
    data = BytesIO(b'\x01\x02\x41\x42\x43')
    parser = BlockParser(
        'root',
        [
            IntegerParser('position', 1),
            IntegerParser('size', 1),
            BackFoldingParser(
                'data',
                [
                    StringParser('string', size=['_parent', 'size'], position=['_parent', 'position'])
                ]
            ),
            StringParser('endString', size=1),
            EOFParser()
        ]
    )

    parser(data)

    assert parser['position'] == 1
    assert parser['size'] == 2
    assert parser['data', 'string'] == 'AB'

    output = BytesIO()

    parser['data', 'string'] = 'ABD'

    parser.unparse(output)
    output.seek(0)
    print(output.read())

    output.close()
    data.close()

def test_new():
    data = b'\x00\x03\x00\x05\x00\xAE\xAA\xA1\x00\xBB'
    parser = BlockParser(
        'base',
        [
            IntegerParser('size', size=2, little_endian=False),
            IntegerParser('position', size=2, little_endian=False),
            Parser('content', 'size', 'position'),
            # Parser('last', position=6),
            # EOFParser()
        ]
    )

    with BytesIO(data) as buffer:
        parser(buffer)

    with BytesIO() as buffer:
        parser.unparse(buffer)
        buffer.seek(0)
        print(buffer.read())

    # with BytesIO() as buffer:
    #     parser.unparse(buffer)
    #     buffer.seek(0)
    #     print(buffer.read())


    # assert parser['size'] == 3
    # assert parser['content'] == b'\xAE\xAA\xA1'
    # assert parser['last'] == b'\xBB'
    # assert parser.getReference('last').getPosition() == 5

    # parser['content'] = b'\xAE\xAA\xA1\xAA'

    # assert parser['size'] == 4
    # assert parser['content'] == b'\xAE\xAA\xA1\xAA'
    # assert parser['last'] == b'\xBB'
    # assert parser.getReference('last').getPosition() == 6
