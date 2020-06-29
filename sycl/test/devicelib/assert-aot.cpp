// REQUIRES: opencl-aot, cpu, linux

// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice %S/assert.cpp %sycl_libs_dir/libsycl-glibc.o %sycl_libs_dir/libsycl-fallback-cassert.o -o %t.aot.out
// RUN: %CPU_RUN_PLACEHOLDER %t.aot.out >%t.aot.msg
// RUN: FileCheck %S/assert.cpp --input-file %t.aot.msg --check-prefixes=CHECK-MESSAGE
