///
/// Perform several driver tests for SYCL device side sanitizers on Linux
///

// UNSUPPORTED: system-windows

/// ###########################################################################

// RUN: %clangxx -fsycl -fsanitize=address -c %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SYCL-ASAN %s
// SYCL-ASAN: clang{{.*}} "-fsycl-is-device"
// SYCL-ASAN-SAME: -fsanitize=address
// SYCL-ASAN-SAME: -fsanitize-address-use-after-return=never
// SYCL-ASAN-SAME: -fno-sanitize-address-use-after-scope
// SYCL-ASAN-SAME: "-mllvm" "-asan-instrumentation-with-call-threshold=0"
// SYCL-ASAN-SAME: "-mllvm" "-asan-constructor-kind=none"
// SYCL-ASAN-SAME: "-mllvm" "-asan-stack=0"
// SYCL-ASAN-SAME: "-mllvm" "-asan-globals=0"
// SYCL-ASAN-SAME: "-mllvm" "-asan-stack-dynamic-alloca=0"
// SYCL-ASAN-SAME: "-mllvm" "-asan-use-after-return=never"
// SYCL-ASAN-SAME: "-mllvm" "-asan-mapping-scale=4"

// RUN: %clangxx -fsycl -Xarch_device -fsanitize=address -c %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SYCL-ASAN-XARCH-DEVICE %s
// SYCL-ASAN-XARCH-DEVICE: clang{{.*}} "-fsycl-is-device"
// SYCL-ASAN-XARCH-DEVICE-SAME: -fsanitize=address

// RUN: %clangxx -fsycl -Xarch_device -fsanitize=address -Xarch_device -fsanitize-recover=address -c %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SYCL-ASAN-RECOVER %s
// SYCL-ASAN-RECOVER: clang{{.*}} "-fsycl-is-device"
// SYCL-ASAN-RECOVER-SAME: -fsanitize=address
// SYCL-ASAN-RECOVER-SAME: -fsanitize-recover=address

/// Make sure "-asan-stack" is always disabled
///
// RUN: %clangxx -fsycl -fsanitize=address -mllvm -asan-stack=1 -c %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SYCL-ASAN-FILTER %s
// SYCL-ASAN-FILTER: clang{{.*}} "-fsycl-is-device"
// SYCL-ASAN-FILTER-SAME: -fsanitize=address
// SYCL-ASAN-FILTER-SAME: "-mllvm" "-asan-stack=0"

/// ###########################################################################

// RUN: %clangxx -fsycl -fsanitize=memory -c %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SYCL-MSAN %s
// SYCL-MSAN: clang{{.*}} "-fsycl-is-device"
// SYCL-MSAN-SAME: -fsanitize=memory
// SYCL-MSAN-SAME: "-mllvm" "-msan-instrumentation-with-call-threshold=0"
// SYCL-MSAN-SAME: "-mllvm" "-msan-eager-checks=1"

// RUN: %clangxx -fsycl -Xarch_device -fsanitize=memory -c %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SYCL-MSAN-XARCH-DEVICE %s
// SYCL-MSAN-XARCH-DEVICE: clang{{.*}} "-fsycl-is-device"
// SYCL-MSAN-XARCH-DEVICE-SAME: -fsanitize=memory

/// ###########################################################################

// RUN: %clangxx -fsycl -fsanitize=thread -c %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SYCL-TSAN %s
// SYCL-TSAN: clang{{.*}} "-fsycl-is-device"
// SYCL-TSAN-SAME: -fsanitize=thread
// SYCL-TSAN-SAME: "-mllvm" "-tsan-instrument-func-entry-exit=0"
// SYCL-TSAN-SAME: "-mllvm" "-tsan-instrument-memintrinsics=0"

// RUN: %clangxx -fsycl -Xarch_device -fsanitize=thread -c %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SYCL-TSAN-XARCH-DEVICE %s
// SYCL-TSAN-XARCH-DEVICE: clang{{.*}} "-fsycl-is-device"
// SYCL-TSAN-XARCH-DEVICE-SAME: -fsanitize=thread
