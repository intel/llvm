// SYCL offloading tests using -fsycl-dump-device-code

// Verify that -fsycl-dump-device-code passes the option to
// clang-linker-wrapper in the new offload model.

<<<<<<<< HEAD:clang/test/Driver/sycl-dump-device-code-old-model.cpp
// Linux
// clang -fsycl --no-offload-new-driver -target x86_64-unknown-linux-gnu
// RUN: %clang -fsycl --no-offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -target x86_64-unknown-linux-gnu -fsycl-dump-device-code=/user/input/path %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHK-FSYCL-DUMP-DEVICE-CODE

// clang -fsycl --no-offload-new-driver -fsycl-targets=spir64-unknown-unknown
// RUN: %clang -fsycl --no-offload-new-driver  -fsycl-targets=spir64-unknown-unknown -target x86_64-unknown-linux-gnu -fsycl-dump-device-code=/user/input/path %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHK-FSYCL-DUMP-DEVICE-CODE

// clang --driver-mode=g++
// RUN: %clangxx -fsycl --no-offload-new-driver  -fsycl-targets=spir64-unknown-unknown -target x86_64-unknown-linux-gnu -fsycl-dump-device-code=/user/input/path %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHK-FSYCL-DUMP-DEVICE-CODE

// Windows
// RUN: %clang_cl -fsycl --no-offload-new-driver -fsycl-dump-device-code=/user/input/path %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHK-FSYCL-DUMP-DEVICE-CODE-WIN
========
// clang -fsycl --offload-new-driver -target x86_64-unknown-linux-gnu
// RUN: %clang -fsycl --offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -target x86_64-unknown-linux-gnu -fsycl-dump-device-code=/user/input/path %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHK-FSYCL-DUMP-DEVICE-CODE-NEW-OFFLOAD

// clang -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown
// RUN: %clang -fsycl --offload-new-driver  -fsycl-targets=spir64-unknown-unknown -target x86_64-unknown-linux-gnu -fsycl-dump-device-code=/user/input/path %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHK-FSYCL-DUMP-DEVICE-CODE-NEW-OFFLOAD

// clang --driver-mode=g++
// RUN: %clangxx -fsycl --offload-new-driver  -fsycl-targets=spir64-unknown-unknown -target x86_64-unknown-linux-gnu -fsycl-dump-device-code=/user/input/path %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHK-FSYCL-DUMP-DEVICE-CODE-NEW-OFFLOAD

// Windows
// RUN: %clang_cl -fsycl --offload-new-driver -fsycl-dump-device-code=/user/input/path %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHK-FSYCL-DUMP-DEVICE-CODE-WIN-NEW-OFFLOAD
>>>>>>>> my_remote/sycl:clang/test/Driver/sycl-dump-device-code.cpp

// CHK-FSYCL-DUMP-DEVICE-CODE-NEW-OFFLOAD: clang-linker-wrapper{{.*}} "-sycl-dump-device-code=/user/input/path"
// CHK-FSYCL-DUMP-DEVICE-CODE-WIN-NEW-OFFLOAD: clang-linker-wrapper{{.*}} "-sycl-dump-device-code=/user/input/path"

// Linux
<<<<<<<< HEAD:clang/test/Driver/sycl-dump-device-code-old-model.cpp
// RUN: %clang -fsycl --no-offload-new-driver  -fsycl-targets=spir64-unknown-unknown -target x86_64-unknown-linux-gnu -fsycl-dump-device-code= %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHK-FSYCL-DUMP-DEVICE-CODE-CWD

// Windows
// RUN: %clang_cl -fsycl --no-offload-new-driver -fsycl-dump-device-code= %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHK-FSYCL-DUMP-DEVICE-CODE-WIN-CWD
========
// RUN: %clang -fsycl --offload-new-driver  -fsycl-targets=spir64-unknown-unknown -target x86_64-unknown-linux-gnu -fsycl-dump-device-code= %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHK-FSYCL-DUMP-DEVICE-CODE-CWD-NEW-OFFLOAD

// Windows
// RUN: %clang_cl -fsycl --offload-new-driver -fsycl-dump-device-code= %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHK-FSYCL-DUMP-DEVICE-CODE-WIN-CWD-NEW-OFFLOAD
>>>>>>>> my_remote/sycl:clang/test/Driver/sycl-dump-device-code.cpp

// CHK-FSYCL-DUMP-DEVICE-CODE-CWD-NEW-OFFLOAD: clang-linker-wrapper{{.*}} "-sycl-dump-device-code="
// CHK-FSYCL-DUMP-DEVICE-CODE-WIN-CWD-NEW-OFFLOAD: clang-linker-wrapper{{.*}} "-sycl-dump-device-code="
