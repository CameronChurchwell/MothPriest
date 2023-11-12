from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from _typeshed import ReadableBuffer
else:
    ReadableBuffer = None
from io import BytesIO

class OffsetBuffer(BytesIO):

    def __init__(self, offset: int = 0) -> None:
        super().__init__()
        self.offset = offset

    def tell(self) -> int:
        return super().tell() + self.offset
    
    def seek(self, __offset: int, __whence: int = 0) -> int:
        if __offset < self.offset:
            raise IOError('Cannot seek before the offset of an OffsetBuffer object')
        return super().seek(__offset - self.offset, __whence)