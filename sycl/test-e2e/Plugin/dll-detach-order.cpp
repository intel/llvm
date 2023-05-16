// REQUIRES: windows
// RUN: env SYCL_PI_TRACE=2 sycl-ls | FileCheck %s

// ensure that the plugins are detached AFTER piTearDown is done executing

// CHECK: ---> DLL_PROCESS_DETACH syclx.dll
// CHECK: ---> piTearDown(

// whatever plugin THIS is
// CHECK: ---> DLL_PROCESS_DETACH

// CHECK: ---> DLL_PROCESS_DETACH win_proxy_loader.dll
