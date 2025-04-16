# Copyright (C) 2022-2024 Intel Corporation
#
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil
import re
import configparser
import glob
import json
import yaml
import subprocess
from mako.template import Template
from mako import exceptions

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def exists(path):
    """safely checks if path/file exists"""
    if path and os.path.exists(path):
        return True
    else:
        return False


def makePath(path):
    """create path if it doesn't exist"""
    try:
        if not exists(path):
            os.makedirs(path)
    except:
        print("warning: failed to make %s" % path)


def copyTree(src, dst):
    """copy tree"""
    try:
        shutil.copytree(src, dst)
    except:
        print("warning: failed to copy %s to %s" % (src, dst))


def removePath(path):
    """remove directory and all contents"""
    try:
        shutil.rmtree(path)
    except:
        print("warning: failed to remove %s" % path)


def removeFile(lst):
    """removes all files in list"""
    for f in lst or []:
        try:
            os.remove(f)
        except:
            print("warning: failed to remove %s" % f)


def findFiles(path, pattern):
    """returns a list of files in path matching pattern"""
    try:
        return sorted(glob.glob(os.path.join(path, pattern)))
    except:
        print("warning: unable to find %s" % path)
        return []


def removeFiles(path, pattern):
    """removes all files in path matching pattern"""
    for f in findFiles(path, pattern):
        try:
            os.remove(f)
        except:
            print("warning: failed to remove %s" % f)


def textRead(path):
    """reads from text file, returns list of lines"""
    try:
        with open(path, "r") as fin:
            return fin.readlines()
    except Exception as e:
        print(e)
        print("error: unable to read %s" % path)
        return None


def configRead(path):
    """read from ini file, returns config obj"""
    try:
        parser = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation()
        )
        parser.read(path)
        return parser
    except:
        print("error: unable to read %s" % path)
        return None


def jsonRead(path):
    """read from json file, returns list/dict"""
    try:
        with open(path, "r") as fin:
            return json.loads(fin.read())
    except:
        print("error: unable to read %s" % path)
        return None


def jsonWrite(path, data):
    """writes list/dict to json file"""
    try:
        with open(path, "w") as fout:
            fout.write(json.dumps(data, indent=2, sort_keys=True))
    except:
        print("error: unable to write %s" % path)


def yamlRead(path):
    """read from yml file, returns list/dict"""
    try:
        with open(path, "r") as fin:
            return yaml.load_all(fin.read(), Loader=Loader)
    except:
        print("error: unable to read %s" % path)
        return None


makoFileList = []
makoErrorList = []


def makoWrite(inpath, outpath, **args):
    """generates file using template, args"""
    try:
        template = Template(filename=inpath)
        rendered = template.render(**args)
        rendered = re.sub(r"\r\n", r"\n", rendered)

        with open(outpath, "w") as fout:
            fout.write(rendered)

        makoFileList.append(outpath)
        return len(rendered.splitlines())
    except:
        print(exceptions.text_error_template().render())
        raise


def makoFileListWrite(outpath):
    jsonWrite(outpath, makoFileList)


def formatGeneratedFiles(clang_format):
    for file in makoFileList:
        if re.search(r"(\.h|\.hpp|\.c|\.cpp|\.def)$", file) is None:
            continue
        print("Formatting {}".format(file))
        proc = subprocess.run(
            [clang_format, "--style=file", "-i", file],
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            print("-- clang-format failed with non-zero return code. --")
            print(proc.stderr.decode())
            raise Exception("Failed to format {}".format(file))


def makeErrorCount():
    return len(makoErrorList)


def writelines(fout, lines):
    """write to array of string lines to file"""
    try:
        with open(fout, "w") as f:
            f.writelines(lines)
            f.close()
    except:
        print("Could not write %s" % fout)
        return None


def to_snake_case(str):
    f = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", str)
    return re.sub("([a-z])([A-Z0-9])", r"\1_\2", f)
