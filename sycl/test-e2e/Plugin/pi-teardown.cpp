// ensure that piTearDown is called

// RUN: env SYCL_PI_TRACE=2 sycl-ls | FileCheck %s
// CHECK: ---> piTearDown
