// Only L0 does adapter release on Windows
// UNSUPPORTED: windows && (opencl || cuda)

// ensure that urAdapterRelease is called

// RUN: env SYCL_UR_TRACE=2 %{run-unfiltered-devices} sycl-ls 2>&1| FileCheck %s
// CHECK: <--- urAdapterRelease
