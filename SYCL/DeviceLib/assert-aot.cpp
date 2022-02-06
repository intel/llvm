// REQUIRES: opencl-aot, cpu, linux

// RUN: %clangxx -DSYCL_FALLBACK_ASSERT=1 -fsycl -fsycl-targets=spir64_x86_64 %S/assert.cpp -o %t.aot.out
// RUN: %CPU_RUN_PLACEHOLDER EXPECTED_SIGNAL=SIGABRT SHOULD_CRASH=1 %t.aot.out 2>%t.aot.msg
// RUN: FileCheck %S/assert.cpp --input-file %t.aot.msg --check-prefixes=CHECK-MESSAGE
