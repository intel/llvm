#!/bin/bash

cmake --build $GITHUB_WORKSPACE/build --target utils/FileCheck/install
cmake --build $GITHUB_WORKSPACE/build --target utils/count/install
cmake --build $GITHUB_WORKSPACE/build --target utils/not/install
cmake --build $GITHUB_WORKSPACE/build --target utils/lit/install
cmake --build $GITHUB_WORKSPACE/build --target utils/llvm-lit/install
cmake --build $GITHUB_WORKSPACE/build --target install-llvm-size
cmake --build $GITHUB_WORKSPACE/build --target install-llvm-cov
cmake --build $GITHUB_WORKSPACE/build --target install-llvm-profdata
cmake --build $GITHUB_WORKSPACE/build --target install-compiler-rt
# This is required to perform the DeviceConfigFile consistency test, see
# sycl/test-e2e/Basic/device_config_file_consistency.cpp.
cmake --install $GITHUB_WORKSPACE/build --component DeviceConfigFile
