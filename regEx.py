import re

data = """
871020-1234567
001225-2345678
831111-3456789
"""

pat = re.compile(r'(\d{6})[-]\d{7}')
print(pat.sub(r'\g<1>-*******', data))