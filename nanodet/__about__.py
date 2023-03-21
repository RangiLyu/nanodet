import time

_this_year = time.strftime("%Y")
__version__ = "1.0.0"
__author__ = "RangiLyu"
__author_email__ = "lyuchqi@gmail.com"
__license__ = "Apache-2.0"
__copyright__ = f"Copyright (c) 2020-{_this_year}, {__author__}."
__homepage__ = "https://github.com/RangiLyu/nanodet"

__docs__ = (
    "NanoDet: Deep learning object detection toolbox for super fast and "
    "lightweight anchor-free object detection models."
)

__all__ = [
    "__author__",
    "__author_email__",
    "__copyright__",
    "__docs__",
    "__homepage__",
    "__license__",
    "__version__",
]
