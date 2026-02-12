// ensure that urAdapterRelease is called

// RUN: env SYCL_UR_TRACE=2 %{run-unfiltered-devices} sycl-ls 2>&1| FileCheck %s
// CHECK: <--- urAdapterRelease
