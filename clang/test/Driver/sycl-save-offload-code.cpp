// SYCL offloading tests using -save-offload-code

// Verify that -save-offload-code passes the option to
// clang-linker-wrapper in the new offload model.

// clang -fsycl --offload-new-driver -target x86_64-unknown-linux-gnu
// RUN: %clang -fsycl --offload-new-driver -fno-sycl-instrument-device-code --no-offloadlib -target x86_64-unknown-linux-gnu -save-offload-code=/user/input/path %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHK-save-offload-code-NEW-OFFLOAD

// clang -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown
// RUN: %clang -fsycl --offload-new-driver  -fsycl-targets=spir64-unknown-unknown -target x86_64-unknown-linux-gnu -save-offload-code=/user/input/path %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHK-save-offload-code-NEW-OFFLOAD

// clang --driver-mode=g++
// RUN: %clangxx -fsycl --offload-new-driver  -fsycl-targets=spir64-unknown-unknown -target x86_64-unknown-linux-gnu -save-offload-code=/user/input/path %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHK-save-offload-code-NEW-OFFLOAD

// Windows
// RUN: %clang_cl -fsycl --offload-new-driver -Qsave-offload-code=/user/input/path %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHK-save-offload-code-WIN-NEW-OFFLOAD

// CHK-save-offload-code-NEW-OFFLOAD: clang-linker-wrapper{{.*}} "-sycl-dump-device-code=/user/input/path"
// CHK-save-offload-code-WIN-NEW-OFFLOAD: clang-linker-wrapper{{.*}} "-sycl-dump-device-code=/user/input/path"

// Linux
// RUN: %clang -fsycl --offload-new-driver  -fsycl-targets=spir64-unknown-unknown -target x86_64-unknown-linux-gnu -save-offload-code= %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHK-save-offload-code-CWD-NEW-OFFLOAD

// Windows
// RUN: %clang_cl -fsycl --offload-new-driver -Qsave-offload-code= %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHK-save-offload-code-WIN-CWD-NEW-OFFLOAD

// CHK-save-offload-code-CWD-NEW-OFFLOAD: clang-linker-wrapper{{.*}} "-sycl-dump-device-code="
// CHK-save-offload-code-WIN-CWD-NEW-OFFLOAD: clang-linker-wrapper{{.*}} "-sycl-dump-device-code="
