"""
Copyright (C) 2025 Intel Corporation

Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE.TXT
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""

import lit.formats
import re
import sys
from os import path

# UR can be built as either a part of intel/llvm or standalone
# The ur_* configs will always refer to directories containing for UR
# The main_* configs will refer to the main cmake directories. If this is built as part of intel/llvm, these will be the
# intel/llvm directories. If built standalone, this will be equal to the ur_* directories.

config.name = "Unified Runtime"
config.test_source_root = path.dirname(__file__)
config.test_exec_root = path.join(config.ur_obj_root, "test")

# Default test configuration - unit tests (that use formats.GoogleTest) use a different test suite specified by
# lit.cfg.py (which does not inherit from this one)
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".test"]


# This is similar to ToolSubst in LLVM, but much simpler
def word_match(key, subst):
    regex = re.compile(rf"\b{key}\b")
    return (regex, subst)


config.substitutions.append((r"%lib", config.main_lib_dir))
config.substitutions.append((r"%binary-dir", config.ur_obj_root))

if sys.platform == "win32":
    config.substitutions.append((r"%null", "NUL"))
else:
    config.substitutions.append((r"%null", "/dev/null"))

if sys.platform == "win32":
    config.shlibext = ".dll"
    config.shlibpre = ""
    config.available_features.add("windows")
if sys.platform == "linux":
    config.shlibext = ".so"
    config.shlibpre = "lib"
    config.available_features.add("linux")
if sys.platform == "darwin":
    config.shlibext = ".dylib"
    config.shlibpre = "lib"

config.substitutions.append((r"%{shlibpre}", config.shlibpre))
config.substitutions.append((r"%{shlibext}", config.shlibext))

# Each adapter provides a substitution like `%use-opencl` which expands to the appropriate "FORCE_LOAD" var
config.adapters = dict()
for a in config.adapters_built:
    config.adapters[a] = path.join(
        config.main_lib_dir, f"{config.shlibpre}ur_adapter_{a}{config.shlibext}"
    )
    config.substitutions.append((f"%adapter-{a}".format(a), config.adapters[a]))
    config.substitutions.append(
        (f"%use-{a}".format(a), f"UR_ADAPTERS_FORCE_LOAD={config.adapters[a]}")
    )
    config.available_features.add(f"adapter-{a}")

# If no adapters are built, don't include the conformance tests
if config.adapters_built == ["mock"]:
    config.excludes.add("conformance")

config.substitutions.append(word_match("FileCheck", config.filecheck_path))


# Ensure built binaries/libs are available on the path
def path_prepend(envvar, value):
    oldvar = config.environment.get(envvar, "")
    config.environment[envvar] = f"{value}{path.pathsep}{oldvar}"


path_prepend("PATH", ".")
path_prepend("PATH", path.join(config.ur_obj_root, "bin"))
if config.main_obj_root != config.ur_obj_root:
    path_prepend("PATH", path.join(config.main_obj_root, "bin"))

if sys.platform == "win32":
    path_prepend("PATH", config.main_lib_dir)
else:
    path_prepend("LD_LIBRARY_PATH", config.main_lib_dir)

config.substitutions.append((r"%ur-version", config.ur_version))
if config.fuzztesting_enabled:
    config.available_features.add("fuzztesting")
if config.tracing_enabled:
    config.available_features.add("tracing")
if config.sanitizer_enabled:
    config.available_features.add("sanitizer")
