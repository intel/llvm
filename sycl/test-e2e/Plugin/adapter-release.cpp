// ensure that urAdapterRelease is called

// RUN: env SYCL_UR_TRACE=2 sycl-ls | FileCheck %s
// CHECK: <--- urAdapterRelease
