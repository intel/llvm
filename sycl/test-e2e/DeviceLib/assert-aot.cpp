// REQUIRES: opencl-aot, cpu, linux

// RUN: %clangxx -DSYCL_FALLBACK_ASSERT=1 -fsycl -fsycl-targets=spir64_x86_64 %S/assert.cpp -o %t.aot.out
// RUN: env EXPECTED_SIGNAL=SIGABRT SHOULD_CRASH=1 %{run} %t.aot.out 2>&1 | FileCheck %S/assert.cpp --check-prefixes=CHECK-MESSAGE
