# Copyright (C) 2024-2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
from utils.utils import download, run
from options import options
import os
import hashlib
import glob


class OneAPI:
    def __init__(self):
        self.oneapi_dir = os.path.join(options.workdir, "oneapi")
        Path(self.oneapi_dir).mkdir(parents=True, exist_ok=True)
        self.oneapi_instance_id = self.generate_unique_oneapi_id(self.oneapi_dir)

        self.install_package(
            "base",
            "2025.1.0+627",
            "https://registrationcenter-download.intel.com/akdlm/IRC_NAS/cca951e1-31e7-485e-b300-fe7627cb8c08/intel-oneapi-base-toolkit-2025.1.0.651_offline.sh",
            "98cad2489f2c90a2b328568a59371cf35855a3338643f61a9fc2d16a265d29f22feb2d673916dd7be18fa12a5e6d2475",
        )
        return

    def generate_unique_oneapi_id(self, path):
        hash_object = hashlib.md5(path.encode())
        return hash_object.hexdigest()

    def check_install(self, version):
        logs_dir = os.path.join(self.oneapi_dir, "logs")
        pattern = f"{logs_dir}/installer.install.intel.oneapi.lin.basekit.product,v={version}*.log"
        log_files = glob.glob(pattern)
        success_line = f"Operation 'intel.oneapi.lin.basekit.product,v={version}' execution is finished with status Success."
        for log_file in log_files:
            try:
                with open(log_file, "r") as f:
                    for line in f:
                        if success_line in line:
                            return True
            except Exception:
                continue
        return False

    def install_package(self, name, version, url, checksum):
        if self.check_install(version):
            print(f"{name} version {version} already installed, skipping.")
            return
        package_name = f"package_{name}.sh"
        package_path = os.path.join(self.oneapi_dir, f"{package_name}")
        if Path(package_path).exists():
            print(f"{package_path} exists, skipping download of oneAPI package...")
        else:
            package = download(
                self.oneapi_dir, url, f"{package_name}", checksum=checksum
            )
        try:
            run(
                f"sh {package_path} -a -s --eula accept --install-dir {self.oneapi_dir} --instance {self.oneapi_instance_id}"
            )
        except:
            print("oneAPI installation likely exists already")
            return
        print(f"{name} installation complete")

    def package_dir(self, package, dir):
        return os.path.join(self.oneapi_dir, package, "latest", dir)

    def package_cmake(self, package):
        package_lib = self.package_dir(package, "lib")
        return os.path.join(package_lib, "cmake", package)

    def mkl_dir(self):
        return self.package_dir("mkl", "")

    def mkl_lib(self):
        return self.package_dir("mkl", "lib")

    def mkl_include(self):
        return self.package_dir("mkl", "include")

    def mkl_cmake(self):
        return self.package_cmake("mkl")

    def dnn_lib(self):
        return self.package_dir("dnnl", "lib")

    def dnn_include(self):
        return self.package_dir("dnnl", "include")

    def dnn_cmake(self):
        return self.package_cmake("dnnl")

    def tbb_lib(self):
        return self.package_dir("tbb", "lib")

    def tbb_cmake(self):
        return self.package_cmake("tbb")

    def compiler_lib(self):
        return self.package_dir("compiler", "lib")

    def ld_libraries(self):
        return [self.compiler_lib(), self.mkl_lib(), self.tbb_lib(), self.dnn_lib()]


oneapi_instance = None


def get_oneapi() -> OneAPI:  # oneAPI singleton
    if not hasattr(get_oneapi, "instance"):
        get_oneapi.instance = OneAPI()
    return get_oneapi.instance
