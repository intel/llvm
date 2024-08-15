// REQUIRES: windows
// RUN: env SYCL_UR_TRACE=-1 sycl-ls | FileCheck %s

// ensure that the plugins are detached AFTER urLoaderTearDown is done executing

// CHECK: ---> DLL_PROCESS_DETACH syclx.dll

// whatever adapter THIS is
// CHECK: ---> urAdapterRelease(
// CHECK: <LOADER>[INFO]: unloaded adapter

// CHECK: ---> urLoaderTearDown(
