"""
 Copyright (C) 2022-2024 Intel Corporation

 Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 See LICENSE.TXT
 SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
import os
import shutil
import re
import configparser
import glob
import json
import yaml
from mako.template import Template
from mako import exceptions
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

"""
    safely checks if path/file exists
"""
def exists(path):
    if path and os.path.exists(path):
        return True
    else:
        return False

"""
    create path if it doesn't exist
"""
def makePath(path):
    try:
        if not exists(path):
            os.makedirs(path)
    except:
        print("warning: failed to make %s"%path)

    
"""
    copy tree
"""
def copyTree(src, dst):
    try:
        shutil.copytree(src, dst)
    except:
        print("warning: failed to copy %s to %s"%(src,dst))

"""
    remove directory and all contents
"""
def removePath(path):
    try:
        shutil.rmtree(path)
    except:
        print("warning: failed to remove %s"%path)

"""
    removes all files in list
"""
def removeFile(lst):
    for f in lst or []:
        try:
            os.remove(f)
        except:
            print("warning: failed to remove %s"%f)

"""
    returns a list of files in path matching pattern
"""
def findFiles(path, pattern):
    try:
        return sorted(glob.glob(os.path.join(path, pattern)))
    except:
        print("warning: unable to find %s"%path)
        return []

"""
    removes all files in path matching pattern
"""
def removeFiles(path, pattern):
    for f in findFiles(path, pattern):
        try:
            os.remove(f)
        except:
            print("warning: failed to remove %s"%f)

"""
    reads from text file, returns list of lines
"""
def textRead(path):
    try:
        with open(path, "r") as fin:
            return fin.readlines()
    except Exception as e:
        print(e)
        print("error: unable to read %s"%path)
        return None

"""
    read from ini file, returns config obj
"""
def configRead(path):
    try:
        parser = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        parser.read(path)
        return parser
    except:
        print("error: unable to read %s"%path)
        return None

"""
    read from json file, returns list/dict
"""
def jsonRead(path):
    try:
        with open(path, 'r') as fin:
            return json.loads(fin.read())
    except:
        print("error: unable to read %s"%path)
        return None

"""
    writes list/dict to json file
"""
def jsonWrite(path, data):
    try:
        with open(path, 'w') as fout:
            fout.write(json.dumps(data, indent=4, sort_keys=True))
    except:
        print("error: unable to write %s"%path)

"""
    read from yml file, returns list/dict
"""
def yamlRead(path):
    try:
        with open(path, 'r') as fin:
            return yaml.load_all(fin.read(), Loader = Loader)
    except:
        print("error: unable to read %s"%path)
        return None

"""
    generates file using template, args
"""
makoFileList = []
makoErrorList = []
def makoWrite(inpath, outpath, **args):
    try:
        template = Template(filename=inpath)
        rendered = template.render(**args)
        rendered = re.sub(r"\r\n", r"\n", rendered)

        with open(outpath, 'w') as fout:
            fout.write(rendered)

        makoFileList.append(outpath)
        return len(rendered.splitlines())
    except:
        print(exceptions.text_error_template().render())
        raise

def makoFileListWrite(outpath):
    jsonWrite(outpath, makoFileList)

def makeErrorCount():
    return len(makoErrorList)

"""
    write to array of string lines to file
"""
def writelines(fout, lines):
    try:
        with open(fout, "w") as f:
            f.writelines(lines)
            f.close()
    except:
        print("Could not write %s"%fout)
        return None

def to_snake_case(str):
    f = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', str)
    return re.sub('([a-z])([A-Z0-9])', r'\1_\2', f)

# END OF FILE
