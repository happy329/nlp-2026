"""Keyword extraction package.

The package name follows the project directory convention.  Python also has a
standard-library module named ``keyword``, so we mirror its public attributes to
avoid surprising third-party imports.
"""

import importlib.util
import os
import sysconfig


_stdlib_keyword_path = os.path.join(sysconfig.get_path("stdlib"), "keyword.py")
_spec = importlib.util.spec_from_file_location("_stdlib_keyword", _stdlib_keyword_path)
_stdlib_keyword = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_stdlib_keyword)

kwlist = _stdlib_keyword.kwlist
softkwlist = getattr(_stdlib_keyword, "softkwlist", [])
iskeyword = _stdlib_keyword.iskeyword
issoftkeyword = getattr(_stdlib_keyword, "issoftkeyword", frozenset(softkwlist).__contains__)

__all__ = ["kwlist", "softkwlist", "iskeyword", "issoftkeyword"]
