"""
Usage: python3 env_config.py
Worksapce: llvm/buildbot

Script to help setup environment alike open source testing. It deals with the environment 
definitions in dependency.conf, linux.vrd and windows.vrd, then it outputs a shell script composed of 
"export" commands. The default name of this shell script is "export.sh" (change in this script as 
needed).
The script "export.sh" is composed of "export" commands, run "source export.sh" to export 
environment variables.
Ensure that the following variables are set correctly before running "source export.sh".

- COMP_ROOT: environment variable pointing to the compiler directory.
- RDRIVE_ROOT: environment variable pointing to the RDRIVE directory.
- ARCHIVE_ROOT: environment variable pointing to the ARCHIVE directory.
"""

import platform
import os
import configparser
import re
import sys

VERSIONS_SECTION = "VERSIONS"
ROOT_SECTION = "DEPS ROOT"

DEPENDENCY_FILE = "dependency.conf"
VRD_FILE_LIN = "linux.vrd"
VRD_FILE_WIN = "windows.vrd"


def filter_options_by_platform(options):
    "filter options according to platform"
    for i in range(len(options)-1, -1, -1):
        is_win_opt = re.search(r'(win|windows)$', options[i], re.I)
        if platform.system() == 'Linux':
            if is_win_opt:
                options.pop(i)
        else:
            if not is_win_opt:
                options.pop(i)


class Dependency():
    "Class to get environment variables according to the defines in dependency.conf and vrd files"
    def __init__(self, deps_version_file):

        self.conf = configparser.ConfigParser()
        self.conf.read(deps_version_file)

        # keys like ['OCLFPGAEMUROOT', 'OCLFPGAROOT', 'OCLCPUROOT', 'OCLGPUROOT', 'TBBROOT', 'OCLOCROOT', 'GCCROOT', ...]
        self.deps_roots = {}
        # keys like ['OCLFPGAEMUVER', 'OCLFPGAVER', 'OCLCPUVER', 'OCLGPUVER', 'TBBVER', 'OCLOCVER', 'GCCVER', ...]
        self.deps_versions = {}

        self.os_system = platform.system()
        self.get_deps_versions()
        self.get_deps_root_path()


    def get_env_from_file(self, vrd_file):
        "Get env var according to vrd file"
        env_dict = {}
        unset_vars = []
        with open(vrd_file, 'r') as f:
            line_list = f.read().splitlines()

        for line in line_list:
            # ignore comments
            if not re.search(r'^:[a-z]+\s+([a-z]|_)+=', line, re.I):
                continue

            fields = line.split()
            fields.pop(0)
            desc = ' '.join(fields)

            vrd = desc.split('=')
            vrd[1] = self.get_completed_env_var(vrd[1])
            if re.search(r'^:ead[eb]', line, re.I):
                if vrd[0] in env_dict:
                    if re.search(r'^:eade', line, re.I):
                        env_dict[vrd[0]] = env_dict[vrd[0]] + os.pathsep + vrd[1]
                    else:
                        env_dict[vrd[0]] = vrd[1] + os.pathsep + env_dict[vrd[0]]
                else:
                    env_dict[vrd[0]] = vrd[1]
            elif re.search(r'^:eset', line, re.I):
                unset_vars.append(vrd[0])
                env_dict[vrd[0]] = vrd[1]
            else:
                pass

        return env_dict, unset_vars


    def get_completed_env_var(self, var):
        "Get completed env var according to XROOT and XVER"
        var = self.get_completed_root(var)

        deps = {'COMP_ROOT': "$COMP_ROOT"}
        deps.update(self.deps_roots)
        deps.update(self.deps_versions)

        for key in deps.keys():
            # ^OCL is the workaround of fpga_ver -> OCLFPGAVER
            pattern = r'{(OCL)*%s}' % key
            value = deps[key].replace('\\', '/')
            var = re.sub(pattern, value, var)

        return var


    def get_deps_versions(self):
        "Get deps version from the VERSION section in dependency.conf"
        options = self.conf.options(VERSIONS_SECTION)
        filter_options_by_platform(options)

        for opt in options:
            value = self.conf.get(VERSIONS_SECTION, opt)

            opt = opt.split('_')
            opt = list(map(lambda x: x.upper(), opt))

            # for 'ocl_cpu_rt_ver', 'ocl_gpu_rt_ver' options, ignore '_rt_' field
            opt = [item for item in opt if item != 'RT']

            opt = ''.join(opt)
            opt = re.sub(r'(WINDOWS|WIN|LIN|LINUX)$', '', opt)

            self.deps_versions[opt] = value


    def get_deps_root_path(self):
        "Get deps version from the ROOT section in dependency.conf"
        options = self.conf.options(ROOT_SECTION)
        filter_options_by_platform(options)

        for opt in options:
            value = self.conf.get(ROOT_SECTION, opt)
            value = self.get_completed_root(value)
            
            opt = opt.split('_')
            opt = list(map(lambda x: x.upper(), opt))
            opt = ''.join(opt)
            opt = re.sub(r'(WINDOWS|WIN|LIN|LINUX)$', '', opt)

            self.deps_roots[opt] = value


    def get_completed_root(self, path):
        "Replace {ARCHIVE_ROOT} and {DEPS_ROOT}"
        new_path = path.replace('{ARCHIVE_ROOT}', "$ARCHIVE_ROOT")
        new_path = new_path.replace('{DEPS_ROOT}', "$RDRIVE_ROOT")
        return new_path


EXPORT_LINE = 'export {}="{}"\n'
env_context = ""
EXPORT_SHELL = "export.sh"


def eset(name, value):
    "Append env var to env_context"
    global env_context
    env_context += (EXPORT_LINE.format(name, value))


def export_file():
    "output export file"
    f = open(EXPORT_SHELL, "w")
    global env_context
    contents = f.write(env_context)
    f.close()


def main():

    deps_version_file = DEPENDENCY_FILE
    vrd_file = VRD_FILE_LIN if platform.system()=='Linux' else VRD_FILE_WIN

    deps = Dependency(deps_version_file)
    env_dict, unset_vars  = deps.get_env_from_file(vrd_file)

    for var in env_dict:
        if var in os.environ and var not in unset_vars:
            value = env_dict[var] + os.pathsep + f"${var}"
        else:
            value = env_dict[var]

        if platform.system()=='Windows':
            value_list = value.split(';')
            for i in range(0, len(value_list)):
                if re.search(':',value_list[i]):
                    tmp = value_list[i].split(':')
                    tmp[0] = '/%s' % tmp[0].lower()
                    tmp[1] = tmp[1].replace('\\','/')
                    value_list[i] = ''.join(tmp)
            value = ':'.join(value_list)
        eset(var, value)

    export_file()

    print(
"""
Open source testing enviroment has been written to "export.sh" successfully. Using the following 
steps to complete the environment setup.
1. Set the following variables correctly.
    - COMP_ROOT: environment variable pointing to the compiler directory.
    - RDRIVE_ROOT: environment variable pointing to the RDRIVE directory.
    - ARCHIVE_ROOT: environment variable pointing to the ARCHIVE directory
2. Run "source export.sh"
"""
    )

    return


if __name__ == "__main__":
    ret = main()
    exit_code = 0 if ret else 1
    sys.exit(exit_code)
