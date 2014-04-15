import sys

PY3 = sys.version_info[0] == 3

if PY3:
    def compat_ord(c):
        return c
else:
    def compat_ord(c):
        return ord(c)
