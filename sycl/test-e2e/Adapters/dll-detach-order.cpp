// REQUIRES: windows
// RUN: env SYCL_UR_TRACE=-1 %{run-unfiltered-devices} sycl-ls | FileCheck %s

// ensure that the adapters are detached AFTER urLoaderTearDown is done
// executing

// CHECK: ---> DLL_PROCESS_DETACH syclx.dll

// whatever adapter THIS is
// CHECK: ---> urAdapterRelease
// CHECK: <LOADER>[INFO]: unloaded adapter

// CHECK: ---> urLoaderTearDown
