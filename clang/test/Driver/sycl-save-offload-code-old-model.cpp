// SYCL offloading tests using -save-offload-code

// Verify that -save-offload-code puts the device code (.spv files)
// in the user provided directory.

// Linux
// clang -fsycl --no-offload-new-driver -target x86_64-unknown-linux-gnu
// RUN: %clang -fsycl --no-offload-new-driver -fno-sycl-instrument-device-code --no-offloadlib -target x86_64-unknown-linux-gnu -save-offload-code=/user/input/path %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHK-save-offload-code

// clang -fsycl --no-offload-new-driver -fsycl-targets=spir64-unknown-unknown
// RUN: %clang -fsycl --no-offload-new-driver  -fsycl-targets=spir64-unknown-unknown -target x86_64-unknown-linux-gnu -save-offload-code=/user/input/path %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHK-save-offload-code

// clang --driver-mode=g++
// RUN: %clangxx -fsycl --no-offload-new-driver  -fsycl-targets=spir64-unknown-unknown -target x86_64-unknown-linux-gnu -save-offload-code=/user/input/path %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHK-save-offload-code

// Windows
// RUN: %clang_cl -fsycl --no-offload-new-driver -Qsave-offload-code=/user/input/path %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHK-save-offload-code-WIN

// CHK-save-offload-code: llvm-foreach{{.*}} "--out-dir=/user/input/path{{(/|\\\\)}}" "--" "{{.*}}llvm-spirv"
// CHK-save-offload-code-WIN: llvm-foreach{{.*}} "--out-dir=/user/input/path{{(/|\\\\)}}" "--" "{{.*}}llvm-spirv"

// Linux
// RUN: %clang -fsycl --no-offload-new-driver  -fsycl-targets=spir64-unknown-unknown -target x86_64-unknown-linux-gnu -save-offload-code= %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHK-save-offload-code-CWD

// Windows
// RUN: %clang_cl -fsycl --no-offload-new-driver -Qsave-offload-code= %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHK-save-offload-code-WIN-CWD

// CHK-save-offload-code-CWD: llvm-foreach{{.*}} "--out-dir=.{{(/|\\\\)}}" "--" "{{.*}}llvm-spirv"
// CHK-save-offload-code-WIN-CWD: llvm-foreach{{.*}} "--out-dir=.{{(/|\\\\)}}" "--" "{{.*}}llvm-spirv"
