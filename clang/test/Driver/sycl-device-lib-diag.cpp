/// Test for SYCL device library diagnostic.

// Only run when libdevice is not enabled.  This allows for a known
// environment that does not have the device libraries installed.
// UNSUPPORTED: libdevice

/// Check for expected device library diagnostic.
// RUN: not %clangxx -fsycl --offload-new-driver %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL-DEVICE-LIB-DIAG
// SYCL-DEVICE-LIB-DIAG: error: cannot find expected SYCL device library 'libsycl-crt.bc'. Pass '--no-offloadlib' to build without the SYCL device libraries

// RUN: %clangxx -fsycl --offload-new-driver %s --sysroot=%S/Inputs/SYCL -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SYCL-DEVICE-LIB-NO-DIAG
// SYCL-DEVICE-LIB-NO-DIAG-NOT: cannot find expected SYCL device library
