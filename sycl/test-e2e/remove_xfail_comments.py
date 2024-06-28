import tempfile
import sys

filepaths = "XFAIL.txt"

filepaths_i = open(filepaths, "r")

for test in filepaths_i:
    test = test[:-1]
    test_i = open(test, "r")
    t = tempfile.NamedTemporaryFile(mode="r+")
    for line in test_i:
        if "XFAIL:" not in line:
            t.write(line)
    test_i.close()
    t.seek(0)
    test_o = open(test, "w")
    for line in t:
        test_o.write(line)
    t.close()
    test_o.close()
