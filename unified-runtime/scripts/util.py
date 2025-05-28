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
from typing import Any, Iterator, List, Union
import yaml
import subprocess
from mako.template import Template
from mako import exceptions

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def exists(path: str) -> bool:
    """safely checks if path/file exists"""
    return bool(path and os.path.exists(path))


def makePath(path: str) -> None:
    """create path if it doesn't exist"""
    try:
        if not exists(path):
            os.makedirs(path)
    except BaseException:
        print("warning: failed to make %s" % path)


def copyTree(src: str, dst: str) -> None:
    """copy tree"""
    try:
        shutil.copytree(src, dst)
    except BaseException:
        print("warning: failed to copy %s to %s" % (src, dst))


def removePath(path: str) -> None:
    """remove directory and all contents"""
    try:
        shutil.rmtree(path)
    except BaseException:
        print("warning: failed to remove %s" % path)


def removeFile(lst: list[str]) -> None:
    """removes all files in list"""
    for f in lst or []:
        try:
            os.remove(f)
        except BaseException:
            print("warning: failed to remove %s" % f)


def findFiles(path: str, pattern: str) -> List[str]:
    """returns a list of files in path matching pattern"""
    try:
        return sorted(glob.glob(os.path.join(path, pattern)))
    except BaseException:
        print("warning: unable to find %s" % path)
        return []


def removeFiles(path: str, pattern: str) -> None:
    """removes all files in path matching pattern"""
    for f in findFiles(path, pattern):
        try:
            os.remove(f)
        except BaseException:
            print("warning: failed to remove %s" % f)


def textRead(path: str) -> List[str]:
    """reads from text file, returns list of lines"""
    try:
        with open(path, "r") as fin:
            return fin.readlines()
    except Exception as e:
        print("error: unable to read %s" % path)
        raise e


def configRead(path: str) -> configparser.ConfigParser:
    """read from ini file, returns config obj"""
    try:
        parser = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation()
        )
        parser.read(path)
        return parser
    except BaseException as error:
        print("error: unable to read %s" % path)
        raise error


def jsonRead(path: str) -> Union[list, dict]:
    """read from json file, returns list/dict"""
    try:
        with open(path, "r") as fin:
            return json.loads(fin.read())
    except BaseException as error:
        print("error: unable to read %s" % path)
        raise error


def jsonWrite(path: str, data: Union[list, dict]) -> None:
    """writes list/dict to json file"""
    try:
        with open(path, "w") as fout:
            fout.write(json.dumps(data, indent=2, sort_keys=True))
    except BaseException as error:
        print("error: unable to write %s" % path)
        raise error


def yamlRead(path: str) -> Iterator[Any]:
    """read from yml file, returns list/dict"""
    try:
        with open(path, "r") as fin:
            return yaml.load_all(fin.read(), Loader=Loader)
    except BaseException as error:
        print("error: unable to read %s" % path)
        raise error


makoFileList = []
makoErrorList = []


def makoWrite(inpath: str, outpath: str, **args) -> int:
    """generates file using template, args"""
    try:
        template = Template(filename=inpath)
        rendered = template.render(**args)
        assert isinstance(rendered, str)
        rendered = re.sub(r"\r\n", r"\n", rendered)

        with open(outpath, "w") as fout:
            fout.write(rendered)

        makoFileList.append(outpath)
        return len(rendered.splitlines())
    except BaseException as error:
        print(exceptions.text_error_template().render())
        raise error


def makoFileListWrite(outpath: str) -> None:
    jsonWrite(outpath, makoFileList)


def formatGeneratedFiles(clang_format: str) -> None:
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


def makeErrorCount() -> int:
    return len(makoErrorList)


def writelines(fout: str, lines: List[str]) -> None:
    """write to array of string lines to file"""
    try:
        with open(fout, "w") as f:
            f.writelines(lines)
            f.close()
    except BaseException:
        print("Could not write %s" % fout)


def to_snake_case(s: str):
    f = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", s)
    return re.sub("([a-z])([A-Z0-9])", r"\1_\2", f)
