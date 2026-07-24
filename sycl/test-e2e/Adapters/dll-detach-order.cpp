// REQUIRES: windows
// RUN: env SYCL_UR_TRACE=-1 %{run-unfiltered-devices} sycl-ls 2>&1 | FileCheck %s

// ensure that the adapters are detached AFTER urLoaderTearDown is done
// executing

// CHECK: ---> DLL_PROCESS_DETACH syclx.dll

// whatever adapter THIS is
// CHECK: ---> urAdapterRelease
// Statically-linked adapters have no library to unload, so the loader's
// "unloaded adapter" message is only emitted for dynamically-loaded adapters.

// CHECK: ---> urLoaderTearDown
