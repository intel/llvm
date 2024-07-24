// REQUIRES: windows
// RUN: env SYCL_UR_TRACE=1 sycl-ls | FileCheck %s

// ensure that the plugins are detached AFTER urLoaderTearDown is done executing

// CHECK: ---> DLL_PROCESS_DETACH syclx.dll
// CHECK: ---> urLoaderTearDown(

// whatever plugin THIS is
// CHECK: ---> DLL_PROCESS_DETACH

// CHECK: ---> DLL_PROCESS_DETACH ur_win_proxy_loader.dll
