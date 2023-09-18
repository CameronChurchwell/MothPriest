from mothpriest.parsers import Parser
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
    parser.__updatePositions()
    assert parser._position == 0