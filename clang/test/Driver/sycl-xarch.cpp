///
/// Perform several driver tests for SYCL -Xarch_device/host on Linux
///

// UNSUPPORTED: system-windows

/// ###########################################################################

/// test behavior of -Xarch_device with 1 option for SYCL compiler, the flag
/// should be passed to device compilation only.
// RUN: %clangxx -fsycl %s -Xarch_device -fsanitize=address -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_XARCH_DEVICE_OPTION
// RUN: %clangxx -fsycl %s -Xarch_device -fsanitize=address -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_XARCH_DEVICE_ONLY
// SYCL_XARCH_DEVICE_OPTION: clang{{.*}} "-fsycl-is-device"
// SYCL_XARCH_DEVICE_OPTION-SAME: -fsanitize=address
// SYCL_XARCH_DEVICE_OPTION-SAME: -fsanitize-address-use-after-return=never
// SYCL_XARCH_DEVICE_OPTION-SAME: -fno-sanitize-address-use-after-scope
// SYCL_XARCH_DEVICE_OPTION-SAME: "-mllvm" "-asan-instrumentation-with-call-threshold=0"
// SYCL_XARCH_DEVICE_OPTION-SAME: "-mllvm" "-asan-stack=0"
// SYCL_XARCH_DEVICE_OPTION-SAME: "-mllvm" "-asan-globals=0"
// SYCL_XARCH_DEVICE_ONLY: llvm-link{{.*}}  "-only-needed"
// SYCL_XARCH_DEVICE_ONLY-NOT: fsanitize=address

/// test behavior of -Xarch_device with multiple options for SYCL compiler, the
/// flags should be passed to device compilation only.
// RUN: %clangxx -fsycl %s -Xarch_device "-fsanitize=address -DXARCH_DEVICE_TEST -mllvm -enable-merge-functions" -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_XARCH_DEVICE_OPTIONS1
// RUN: %clangxx -fsycl %s -Xarch_device "-fsanitize=address -DXARCH_DEVICE_TEST -mllvm -enable-merge-functions" -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_XARCH_DEVICE_OPTIONS1
// RUN: %clangxx -fsycl %s -Xarch_device "-fsanitize=address -DXARCH_DEVICE_TEST -mllvm -enable-merge-functions" -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_XARCH_DEVICE_OPTIONS2
// RUN: %clangxx -fsycl %s -Xarch_device "-fsanitize=address -DXARCH_DEVICE_TEST -mllvm -enable-merge-functions" -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_XARCH_DEVICE_OPTIONS3
// SYCL_XARCH_DEVICE_OPTIONS1: clang{{.*}} "-fsycl-is-device"
// SYCL_XARCH_DEVICE_OPTIONS1-SAME: -fsanitize=address
// SYCL_XARCH_DEVICE_OPTIONS1-SAME: -fsanitize-address-use-after-return=never
// SYCL_XARCH_DEVICE_OPTIONS1-SAME: -fno-sanitize-address-use-after-scope
// SYCL_XARCH_DEVICE_OPTIONS1-SAME: "-mllvm" "-asan-instrumentation-with-call-threshold=0"
// SYCL_XARCH_DEVICE_OPTIONS1-SAME: "-mllvm" "-asan-stack=0"
// SYCL_XARCH_DEVICE_OPTIONS1-SAME: "-mllvm" "-asan-globals=0"
// SYCL_XARCH_DEVICE_OPTIONS2: clang{{.*}} "-fsycl-is-device"
// SYCL_XARCH_DEVICE_OPTIONS2-SAME: XARCH_DEVICE_TEST
// SYCL_XARCH_DEVICE_OPTIONS3: clang{{.*}} "-fsycl-is-device"
// SYCL_XARCH_DEVICE_OPTIONS3-SAME: "-mllvm" "-enable-merge-functions"


/// test behavior of -Xarch_host with 1 option for SYCL compiler, the flag
/// should be passed to host compilation only.
// RUN: %clangxx -fsycl %s -Xarch_host -fsanitize=address -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_XARCH_HOST_OPTION
// RUN: %clangxx -fsycl %s -Xarch_host -fsanitize=address -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_XARCH_HOST_ONLY
// SYCL_XARCH_HOST_OPTION: clang{{.*}} "-fsycl-is-host"
// SYCL_XARCH_HOST_OPTION-SAME: -fsanitize=address
// SYCL_XARCH_HOST_OPTION-SAME: -fsanitize-address-use-after-scope
// SYCL_XARCH_HOST_OPTION-NEXT: libclang_rt.asan
// SYCL_XARCH_HOST_ONLY: clang{{.*}} "-fsycl-is-device"
// SYCL_XARCH_HOST_ONLY-NOT: -fsanitize=address
// SYCL_XARCH_HOST_ONLY: clang{{.*}} "-fsycl-is-host"

/// test behavior of -Xarch_host with multiple options for SYCL compiler, the
/// flags should be passed to host compilation only.
// RUN: %clangxx -fsycl %s -Xarch_host "-fsanitize=address -DXARCH_HOST_TEST -mllvm -enable-merge-functions" -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_XARCH_HOST_OPTIONS1
// RUN: %clangxx -fsycl %s -Xarch_host "-fsanitize=address -DXARCH_HOST_TEST -mllvm -enable-merge-functions" -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_XARCH_HOST_OPTIONS2
// RUN: %clangxx -fsycl %s -Xarch_host "-fsanitize=address -DXARCH_HOST_TEST -mllvm -enable-merge-functions" -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_XARCH_HOST_OPTIONS3
// SYCL_XARCH_HOST_OPTIONS1: clang{{.*}} "-fsycl-is-host"
// SYCL_XARCH_HOST_OPTIONS1-SAME: -fsanitize=address
// SYCL_XARCH_HOST_OPTIONS1-SAME: -fsanitize-address-use-after-scope
// SYCL_XARCH_HOST_OPTIONS2: clang{{.*}} "-fsycl-is-host"
// SYCL_XARCH_HOST_OPTIONS2-SAME: XARCH_HOST_TEST
// SYCL_XARCH_HOST_OPTIONS3: clang{{.*}} "-fsycl-is-host"
// SYCL_XARCH_HOST_OPTIONS3-SAME: "-mllvm" "-enable-merge-functions"

// test behavior of combination of -Xarch_device and -Xarch_device.
// RUN: %clangxx -fsycl %s -Xarch_device "-fsanitize=address -mllvm -enable-merge-functions" \
// RUN:   -Xarch_host "-fsanitize=memory -DUSE_XARCH_HOST -fno-builtin" -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_XARCH_COM_DEVICE_OPTIONS1
// RUN: %clangxx -fsycl %s -Xarch_device "-fsanitize=address -mllvm -enable-merge-functions" \
// RUN:   -Xarch_host "-fsanitize=memory -DUSE_XARCH_HOST -fno-builtin" -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_XARCH_COM_DEVICE_OPTIONS2
// RUN: %clangxx -fsycl %s -Xarch_device "-fsanitize=address -mllvm -enable-merge-functions" \
// RUN:   -Xarch_host "-fsanitize=memory -DUSE_XARCH_HOST -fno-builtin" -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_XARCH_COM_NO_DEVICE
// RUN: %clangxx -fsycl %s -Xarch_device "-fsanitize=address -mllvm -enable-merge-functions" \
// RUN:   -Xarch_host "-fsanitize=memory -DUSE_XARCH_HOST -fno-builtin" -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_XARCH_COM_HOST_OPTIONS1
// RUN: %clangxx -fsycl %s -Xarch_device "-fsanitize=address -mllvm -enable-merge-functions" \
// RUN:   -Xarch_host "-fsanitize=memory -DUSE_XARCH_HOST -fno-builtin" -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_XARCH_COM_HOST_OPTIONS2
// RUN: %clangxx -fsycl %s -Xarch_device "-fsanitize=address -mllvm -enable-merge-functions" \
// RUN:   -Xarch_host "-fsanitize=memory -DUSE_XARCH_HOST -fno-builtin" -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_XARCH_COM_HOST_OPTIONS3
// RUN: %clangxx -fsycl %s -Xarch_device "-fsanitize=address -mllvm -enable-merge-functions" \
// RUN:   -Xarch_host "-fsanitize=memory -DUSE_XARCH_HOST -fno-builtin" -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL_XARCH_COM_NO_HOST
// SYCL_XARCH_COM_DEVICE_OPTIONS1: clang{{.*}} "-fsycl-is-device"
// SYCL_XARCH_COM_DEVICE_OPTIONS1-SAME: -fsanitize=address
// SYCL_XARCH_COM_DEVICE_OPTIONS1-SAME: -fsanitize-address-use-after-return=never
// SYCL_XARCH_COM_DEVICE_OPTIONS1-SAME: -fno-sanitize-address-use-after-scope
// SYCL_XARCH_COM_DEVICE_OPTIONS1-SAME: "-mllvm" "-asan-instrumentation-with-call-threshold=0"
// SYCL_XARCH_COM_DEVICE_OPTIONS1-SAME: "-mllvm" "-asan-stack=0"
// SYCL_XARCH_COM_DEVICE_OPTIONS1-SAME: "-mllvm" "-asan-globals=0"
// SYCL_XARCH_COM_DEVICE_OPTIONS2: clang{{.*}} "-fsycl-is-device"
// SYCL_XARCH_COM_DEVICE_OPTIONS2-SAME: "-mllvm" "-enable-merge-functions"
// SYCL_XARCH_COM_NO_DEVICE: clang{{.*}} "-fsycl-is-device"
// SYCL_XARCH_COM_NO_DEVICE-NOT: USE_XARCH_HOST
// SYCL_XARCH_COM_NO_DEVICE: clang{{.*}} "-fsycl-is-host"
// SYCL_XARCH_COM_HOST_OPTIONS1: clang{{.*}} "-fsycl-is-host"
// SYCL_XARCH_COM_HOST_OPTIONS1-SAME: -fsanitize=memory
// SYCL_XARCH_COM_HOST_OPTIONS1-NEXT: libclang_rt.msan
// SYCL_XARCH_COM_HOST_OPTIONS2: clang{{.*}} "-fsycl-is-host"
// SYCL_XARCH_COM_HOST_OPTIONS2-SAME: USE_XARCH_HOST
// SYCL_XARCH_COM_HOST_OPTIONS3: clang{{.*}} "-fsycl-is-host"
// SYCL_XARCH_COM_HOST_OPTIONS3-SAME: -fno-builtin
// SYCL_XARCH_COM_NO_HOST: clang{{.*}} "-fsycl-is-host"
// SYCL_XARCH_COM_NO_HOST-NOT: "-mllvm" "-enable-merge-functions"
