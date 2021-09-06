// RUN: %clangxx -fsycl -DFAKE_PLUGIN -shared %s -o %t_fake_plugin.so
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env SYCL_OVERRIDE_PI_OPENCL=%t_fake_plugin.so env SYCL_OVERRIDE_PI_LEVEL_ZERO=%t_fake_plugin.so env SYCL_OVERRIDE_PI_CUDA=%t_fake_plugin.so env SYCL_OVERRIDE_PI_ROCM=%t_fake_plugin.so env SYCL_PI_TRACE=-1 %t.out > %t.log 2>&1
// RUN: FileCheck %s --input-file %t.log
// REQUIRES: linux

#ifdef FAKE_PLUGIN

#include <CL/sycl/detail/pi.h>

pi_result piPlatformsGet(pi_uint32 NumEntries, pi_platform *Platforms,
                         pi_uint32 *NumPlatforms) {
  return PI_INVALID_OPERATION;
}

pi_result piTearDown(void *) { return PI_SUCCESS; }

pi_result piPluginInit(pi_plugin *PluginInit) {
  PluginInit->PiFunctionTable.piPlatformsGet = piPlatformsGet;
  PluginInit->PiFunctionTable.piTearDown = piTearDown;
  return PI_SUCCESS;
}

#else

#include <sycl/sycl.hpp>

int main() {
  try {
    sycl::platform P{sycl::default_selector{}};
  } catch (...) {
    // NOP
  }

  return 0;
}

#endif

// CHECK: SYCL_PI_TRACE[basic]: Plugin found and successfully loaded: {{[0-9a-zA-Z_\/\.-]+}}_fake_plugin.so
// CHECK: SYCL_PI_TRACE[basic]: Plugin found and successfully loaded: {{[0-9a-zA-Z_\/\.-]+}}_fake_plugin.so
// CHECK: SYCL_PI_TRACE[basic]: Plugin found and successfully loaded: {{[0-9a-zA-Z_\/\.-]+}}_fake_plugin.so
// CHECK: SYCL_PI_TRACE[basic]: Plugin found and successfully loaded: {{[0-9a-zA-Z_\/\.-]+}}_fake_plugin.so
