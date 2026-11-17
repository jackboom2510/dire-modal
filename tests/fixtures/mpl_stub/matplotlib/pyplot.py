"""`matplotlib.pyplot` giả: mọi hàm vẽ là no-op, `savefig` ghi file rỗng."""

from __future__ import annotations

import pathlib

rcParams: dict = {}


class _Axes:
    def __getattr__(self, name):
        def _noop(*a, **k):
            return None
        return _noop

    def plot(self, *a, **k):
        return [None]

    def scatter(self, *a, **k):
        return None

    def imshow(self, *a, **k):
        return None

    def text(self, *a, **k):
        return None

    def set_yscale(self, *a, **k):
        return None

    def legend(self, *a, **k):
        return None

    def add_patch(self, *a, **k):
        return None

    def get_yaxis_transform(self):
        return None


class _GridSpec:
    def __getitem__(self, item):
        return item


class Figure:
    def __init__(self, *a, **k):
        pass

    def add_subplot(self, *a, **k):
        return _Axes()

    def add_gridspec(self, *a, **k):
        return _GridSpec()

    def add_axes(self, *a, **k):
        return _Axes()

    def subplots_adjust(self, *a, **k):
        return None

    def colorbar(self, *a, **k):
        return None

    def text(self, *a, **k):
        return None

    def suptitle(self, *a, **k):
        return None

    def savefig(self, path, *a, **k):
        pathlib.Path(path).write_bytes(b"%PDF-1.4 stub\n")

    def tight_layout(self, *a, **k):
        return None


class _Subplots:
    """`plt.subplots` trả `(fig, axes)`; `axes` phải index được như ndarray 2 chiều."""

    def __init__(self, nrows, ncols, squeeze=True):
        self.nrows, self.ncols = nrows, ncols
        self._cells = [[_Axes() for _ in range(ncols)] for _ in range(nrows)]
        self.shape = (nrows, ncols)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            return self._cells[item[0]][item[1]]
        return self._cells[0][item]

    def __len__(self):
        return self.nrows

    def __iter__(self):
        for row in self._cells:
            yield from row

    def reshape(self, *shape):
        self.shape = tuple(shape)
        return self

    def ravel(self):
        return list(self)


class Circle:
    def __init__(self, *a, **k):
        pass


class Rectangle:
    def __init__(self, *a, **k):
        pass


def subplots(nrows=1, ncols=1, squeeze=True, **k):
    """Giống matplotlib thật: 1×1 thì trả thẳng `Axes`, không phải mảng.

    Không mô phỏng đúng chỗ này thì test "xanh" trên stub và đỏ trên máy thật —
    loại test tệ nhất.
    """
    if nrows == 1 and ncols == 1 and squeeze:
        return Figure(), _Axes()
    if nrows == 1 or ncols == 1:
        if squeeze:
            return Figure(), _Subplots(nrows, ncols)
        return Figure(), _Subplots(nrows, ncols)     # squeeze=False: vẫn là mảng
    return Figure(), _Subplots(nrows, ncols)


def figure(*a, **k):
    return Figure()


def close(*a, **k):
    return None
