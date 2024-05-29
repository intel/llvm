// ensure that urLoaderTearDown is called

// RUN: env SYCL_UR_TRACE=1 sycl-ls | FileCheck %s
// CHECK: ---> urLoaderTearDown
