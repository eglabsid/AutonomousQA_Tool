
import re
pattern = r"apple|orange"
# pattern = re.escape(name)
# print(pattern)
compiled_pattern = re.compile(pattern)
print(compiled_pattern)
finditer = compiled_pattern.finditer("123 apples_studio and 456 oranges_t-bo")
# finditer = compiled_pattern.findall("123 apples and 456 oranges")
print(finditer)
for match in finditer:
    print("Finditer match:", match.group())
