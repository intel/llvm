// UNSUPPORTED: system-windows

// -print-file-name=libsycl.so
// RUN: %clangxx -print-file-name=libsycl.so --sysroot=%S/Inputs/SYCL 2>&1 | FileCheck %s --check-prefix=PRINT_LIBSYCL
// RUN: %clangxx -fsycl -print-file-name=libsycl.so --sysroot=%S/Inputs/SYCL 2>&1 | FileCheck %s --check-prefix=PRINT_LIBSYCL
// PRINT_LIBSYCL: {{.*}}lib{{(\\\\|/)}}libsycl.so
