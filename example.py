from io import BytesIO
from mothpriest.parsers import *

parser = BlockParser(
    "root",
    [
        IntegerParser('checksum', size=16, little_endian=False),
        IntegrityParser(
            "content",
            [
                MagicParser("STRING", "Magic"),
                IntegerParser("StringSize", size=4, little_endian=True, signed=False),
                StringParser("String", size="StringSize")
            ],
            'md5',
            'checksum'
        )
    ]
)

data = b"\xd2$z(\xaeT2u\xb8\xc6\x16\x9e\xd0;\xe2\x0bSTRING\x1A\x00\x00\x00Lorem ipsum dolor sit amet"
with BytesIO(data) as buffer:
    parser(buffer)

assert parser['content', 'String'] == "Lorem ipsum dolor sit amet"

parser['content', 'String'] = "The quick brown fox jumps over the lazy dog"
# StringSize is now inconsistent with String, but there is no need to manually update

with BytesIO() as buffer:
    parser.unparse(buffer)
    buffer.seek(0)
    output_data = buffer.read()

# Note how StringSize is updated automatically
assert output_data == b"\x1e\x8bNB\xdb\x0f\xdf\xdd\x15g\xd8\x96\xd6\xe3WsSTRING\x2B\x00\x00\x00The quick brown fox jumps over the lazy dog"