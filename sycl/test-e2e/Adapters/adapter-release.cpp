// UNSUPPORTED: windows && (opencl || cuda)
// UNSUPPORTED-INTENDED: Only L0 does adapter release on Windows. Windows
// recommends not doing cleanup at unloading, however L0 still requires this in
// order to enable UR_L0_LEAKS_DEBUG on Windows.

// ensure that urAdapterRelease is called

// RUN: env SYCL_UR_TRACE=2 %{run-unfiltered-devices} sycl-ls 2>&1| FileCheck %s
// CHECK: <--- urAdapterRelease
