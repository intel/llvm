# Copyright (C) 2024-2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
from utils.utils import download, run
from options import options
import os
import hashlib


class OneAPI:
    def __init__(self):
        self.oneapi_dir = os.path.join(options.workdir, "oneapi")
        Path(self.oneapi_dir).mkdir(parents=True, exist_ok=True)
        self.oneapi_instance_id = self.generate_unique_oneapi_id(self.oneapi_dir)

        # can we just hardcode these links?
        self.install_package(
            "dnnl",
            "https://registrationcenter-download.intel.com/akdlm/IRC_NAS/87e117ab-039b-437d-9c80-dcd5c9e675d5/intel-onednn-2025.0.0.862_offline.sh",
            "6866feb5b8dfefd6ff45d6bfabed44f01d7fba8fd452480ae1fd86b92e9481ae052c24842da14f112f672f5c4859945b",
        )
        self.install_package(
            "mkl",
            "https://registrationcenter-download.intel.com/akdlm/IRC_NAS/79153e0f-74d7-45af-b8c2-258941adf58a/intel-onemkl-2025.0.0.940_offline.sh",
            "122bb84cf943ea27753cb399c81ab2ae218ebd51b789c74d273240157722925ab4d5a43cb0b5de41b854f2c5a59a4002",
        )
        return

    def generate_unique_oneapi_id(self, path):
        hash_object = hashlib.md5(path.encode())
        return hash_object.hexdigest()

    def install_package(self, name, url, checksum):
        package_path = os.path.join(self.oneapi_dir, name)
        if Path(package_path).exists():
            print(
                f"{package_path} exists, skipping installing oneAPI package {name}..."
            )
            return

        package = download(
            self.oneapi_dir, url, f"package_{name}.sh", checksum=checksum
        )
        try:
            print(f"installing {name}")
            run(
                f"sh {package} -a -s --eula accept --install-dir {self.oneapi_dir} --instance {self.oneapi_instance_id}"
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
