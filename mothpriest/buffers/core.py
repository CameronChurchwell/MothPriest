from typing import TYPE_CHECKING, Any
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

class DivergentBuffer(BytesIO):

    def __init__(self, original: BytesIO, offset: int=0) -> None:
        super().__init__()
        self.original = original
        self.offset = offset
        # We start in the diverged buffer
        self.diverged = True

    def tell(self) -> int:
        if self.diverged:
            return super().tell() + self.offset
        else:
            position = self.original.tell()
            if position >= self.offset:
                raise IOError("Somehow, DivergentBuffer is pointing past the offset in the original buffer?")
            return position

    def seek(self, __offset: int, __whence: int = 0) -> int:
        assert __whence == 0 #TODO update to properly use whence
        if __offset < self.offset:
            self.diverged = False
            return self.original.seek(__offset, __whence)
        else:
            self.diverged = True
            return super().seek(__offset - self.offset, __whence)

    def write(self, __buffer: ReadableBuffer) -> int:
        if self.diverged:
            return super().write(__buffer)
        else:
            result = self.original.write(__buffer)
            if self.original.tell() > self.offset:
                raise IOError("Buffer wrote into divergent region of original buffer")
            elif self.original.tell() == self.offset: # edge case where we wrote right up to the edge
                pass
                self.diverged = True
                super().seek(self.offset) # probably not necessary, but shouldn't hurt?
            return result

    def read(self, __size: int | None = None) -> bytes:
        if self.diverged:
            return super().read(__size)
        else:
            if not __size <= self.offset - self.original.tell():
                raise IOError("Attempting to read into divergent region of buffer")
            return self.original.read(__size)