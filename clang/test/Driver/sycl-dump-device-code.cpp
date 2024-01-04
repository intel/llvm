// SYCL offloading tests using -fsycl-dump-device-code

// Verify that -fsycl-dump-device-code puts the device code (.spv files)
// in the user provided directory.

// Linux
// clang -fsycl -target x86_64-unknown-linux-gnu
// RUN: %clang -fsycl -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -target x86_64-unknown-linux-gnu -fsycl-dump-device-code=/user/input/path %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHK-FSYCL-DUMP-DEVICE-CODE

// clang -fsycl -fsycl-targets=spir64-unknown-unknown
// RUN: %clang -fsycl  -fsycl-targets=spir64-unknown-unknown -target x86_64-unknown-linux-gnu -fsycl-dump-device-code=/user/input/path %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHK-FSYCL-DUMP-DEVICE-CODE

// clang --driver-mode=g++
// RUN: %clangxx -fsycl  -fsycl-targets=spir64-unknown-unknown -target x86_64-unknown-linux-gnu -fsycl-dump-device-code=/user/input/path %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHK-FSYCL-DUMP-DEVICE-CODE

// Windows
// RUN: %clang_cl -fsycl -fsycl-dump-device-code=/user/input/path %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHK-FSYCL-DUMP-DEVICE-CODE-WIN

// CHK-FSYCL-DUMP-DEVICE-CODE: llvm-foreach{{.*}} "--out-dir=/user/input/path{{(/|\\\\)}}" "--" "{{.*}}llvm-spirv"
// CHK-FSYCL-DUMP-DEVICE-CODE-WIN: llvm-foreach{{.*}} "--out-dir=/user/input/path{{(/|\\\\)}}" "--" "{{.*}}llvm-spirv"

// Linux
// RUN: %clang -fsycl  -fsycl-targets=spir64-unknown-unknown -target x86_64-unknown-linux-gnu -fsycl-dump-device-code= %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHK-FSYCL-DUMP-DEVICE-CODE-CWD

// Windows
// RUN: %clang_cl -fsycl -fsycl-dump-device-code= %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHK-FSYCL-DUMP-DEVICE-CODE-WIN-CWD

// CHK-FSYCL-DUMP-DEVICE-CODE-CWD: llvm-foreach{{.*}} "--out-dir=.{{(/|\\\\)}}" "--" "{{.*}}llvm-spirv"
// CHK-FSYCL-DUMP-DEVICE-CODE-WIN-CWD: llvm-foreach{{.*}} "--out-dir=.{{(/|\\\\)}}" "--" "{{.*}}llvm-spirv"
