// ensure that urAdapterRelease is called

// RUN: env SYCL_UR_TRACE=2 %{run-unfiltered-devices} sycl-ls | FileCheck %s
// CHECK: <--- urAdapterRelease
