///
/// Perform driver test for SYCL RTC mode.
///

/// Check that the '-fsycl-rtc-mode' is correctly forwarded to the device
/// compilation and only to the device compilation.

// RUN: %clangxx -fsycl -fsycl-rtc-mode --no-offload-new-driver %s -### 2>&1 \
// RUN:   | FileCheck %s

// RUN: %clangxx -fsycl -fsycl-rtc-mode --offload-new-driver %s -### 2>&1 \
// RUN:   | FileCheck %s

// CHECK: clang{{.*}} "-fsycl-is-device"
// CHECK-SAME: -fsycl-rtc-mode
// CHECK: clang{{.*}} "-fsycl-is-host"
// CHECK-NOT: -fsycl-rtc-mode


/// Check that the '-fno-sycl-rtc-mode' is correctly forwarded to the device
/// compilation and only to the device compilation.

// RUN: %clangxx -fsycl -fno-sycl-rtc-mode --no-offload-new-driver %s -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NEGATIVE

// RUN: %clangxx -fsycl -fno-sycl-rtc-mode --offload-new-driver %s -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NEGATIVE

// NEGATIVE: clang{{.*}} "-fsycl-is-device"
// NEGATIVE-SAME: -fno-sycl-rtc-mode
// NEGATIVE: clang{{.*}} "-fsycl-is-host"
// NEGATIVE-NOT: -fsycl-rtc-mode
