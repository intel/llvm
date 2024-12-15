///
/// Perform several driver tests for SYCL device side sanitizers on Windows
///

// REQUIRES: system-windows

/// ###########################################################################

// RUN: %clangxx -fsycl -fsanitize=address -c %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SYCL-ASAN %s
// RUN: %clangxx -fsycl -Xarch_device -fsanitize=address -c %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SYCL-ASAN %s

// SYCL-ASAN: ignoring '-fsanitize=address' option as it is not currently supported for target 'spir64-unknown-unknown'

/// ###########################################################################

// We need to add "not" here since "error: unsupported option '-fsanitize=memory' for target 'x86_64-pc-windows-msvc'"
// RUN: not %clangxx -fsycl -fsanitize=memory -c %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SYCL-MSAN %s
// RUN: %clangxx -fsycl -Xarch_device -fsanitize=memory -c %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SYCL-MSAN %s

// SYCL-MSAN: ignoring '-fsanitize=memory' option as it is not currently supported for target 'spir64-unknown-unknown'
