import lit.formats
import sys
from os import path

config.name = "Unified Runtime"
config.test_source_root = path.dirname(__file__)
config.test_exec_root = path.join(config.ur_obj_root, 'test')

uur_path = path.join(config.ur_src_root, "conformance", "testing", "include")

config.substitutions.append((r"%lib", config.root_lib_dir))

if sys.platform == "win32":
    config.shlibext = ".dll"
    config.shlibpre = ""
if sys.platform == "linux":
    config.shlibext = ".so"
    config.shlibpre = "lib"
if sys.platform == "darwin":
    config.shlibext = ".dylib"
    config.shlibpre = ""

config.substitutions.append((r"%{shlibpre}", config.shlibpre))
config.substitutions.append((r"%{shlibext}", config.shlibext))

def make_libname(name):
    return path.join(config.shlibpre + name + config.shlibext)

# Adapter names should be full paths
config.mock_adapter = path.join(config.root_lib_dir, make_libname("ur_adapter_mock"))
config.substitutions.append((r"%mockadapter", config.mock_adapter))

if not config.filecheck_path.endswith("-NOTFOUND") and config.filecheck_path:
    config.substitutions.append((r"%filecheck", config.filecheck_path))
    config.available_features.add("filecheck")

# Ensure built binaries/libs are available on the path
config.environment["PATH"] = path.join(config.ur_obj_root, "bin") + path.pathsep + config.environment.get("PATH", "")
config.environment["PATH"] = path.join(config.root_obj_root, "bin") + path.pathsep + config.environment.get("PATH", "")
if sys.platform == "win32":
    config.environment["PATH"] = config.root_lib_dir + path.pathsep + config.environment.get("PATH", "")
else:
    config.environment["LD_LIBRARY_PATH"] = config.root_lib_dir + path.pathsep + config.environment.get("LD_LIBRARY_PATH", "")
