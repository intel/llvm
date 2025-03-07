"""
Copyright (C) 2025 Intel Corporation

Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE.TXT
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""

import lit.formats
import sys
from os import path

# UR can be built as either a part of intel/dpcpp or standalone
# The ur_* configs will always refer to directories containing for UR
# The main_* configs will refer to the main cmake directories. If this is built as part of intel/llvm, these will be the
# intel/llvm directories. If built standalone, this will be equal to the ur_* directories.

config.name = "Unified Runtime"
config.test_source_root = path.dirname(__file__)
config.test_exec_root = path.join(config.ur_obj_root, "test")

config.substitutions.append((r"%lib", config.main_lib_dir))

if sys.platform == "win32":
    config.shlibext = ".dll"
    config.shlibpre = ""
if sys.platform == "linux":
    config.shlibext = ".so"
    config.shlibpre = "lib"
if sys.platform == "darwin":
    config.shlibext = ".dylib"
    config.shlibpre = "lib"

config.substitutions.append((r"%{shlibpre}", config.shlibpre))
config.substitutions.append((r"%{shlibext}", config.shlibext))

# Each adapter provides a substitution like `%use-opencl` which expands to the appropriate "FORCE_LOAD" var
config.adapters = dict()
for a in ["mock", "level_zero", "opencl", "cuda", "hip", "native_cpu"]:
    config.adapters[a] = path.join(
        config.main_lib_dir, f"{config.shlibpre}ur_adapter_{a}{config.shlibext}"
    )
    config.substitutions.append((f"%adapter-{a}".format(a), config.adapters[a]))
    config.substitutions.append(
        (f"%use-{a}".format(a), f"UR_ADAPTERS_FORCE_LOAD={config.adapters[a]}")
    )

if config.filecheck_path is not None:
    config.substitutions.append((r"%filecheck", config.filecheck_path))
    config.available_features.add("filecheck")


# Ensure built binaries/libs are available on the path
def path_append(envvar, value):
    oldvar = config.environment.get(envvar, "")
    config.environment[envvar] = f"{value}{path.pathsep}{oldvar}"


path_append("PATH", path.join(config.ur_obj_root, "bin"))
if config.main_obj_root != config.ur_obj_root:
    path_append("PATH", path.join(config.main_obj_root, "bin"))

if sys.platform == "win32":
    path_append("PATH", config.main_lib_dir)
else:
    path_append("LD_LIBRARY_PATH", config.main_lib_dir)
