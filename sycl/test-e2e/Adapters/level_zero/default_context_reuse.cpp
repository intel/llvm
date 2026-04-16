// REQUIRES: level_zero_v2_adapter

// API 1.14+ is required for zeDriverGetDefaultContext
// REQUIRES-INTEL-DRIVER: lin: 36300, win: 101.7080

// RUN: %{build} %level_zero_options -o %t.out
// RUN: %{run} %t.out

// Test that full-platform SYCL contexts are reused and match the platform
// default context when all root devices of a platform are requested.
// Verifies URT-1145 behavior from the SYCL observable side.
//
// Before the fix: urContextCreate always calls zeContextCreate, so two
// full-platform contexts have different ze_context handles -> test FAILS.
// After the fix:  urContextCreate reuses the driver default context, so both
// calls return the same handle -> test PASSES.

#include <iostream>
#include <level_zero/ze_api.h>
#include <sycl/backend.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/platform.hpp>
#include <vector>

int main() {
  int failed = 0;
  bool anyTested = false;

  for (auto &platform : sycl::platform::get_platforms()) {
    if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero)
      continue;

    // Collect all root devices exposed by this platform.
    std::vector<sycl::device> rootDevices = platform.get_devices();
    if (rootDevices.empty())
      continue;

    anyTested = true;

    // Create two independent SYCL contexts with the full set of root devices.
    // After the fix both should wrap the same underlying ze_context (the
    // driver default).  Before the fix each call produces a distinct handle.
    sycl::context ctx1(rootDevices);
    sycl::context ctx2(rootDevices);

    ze_context_handle_t h1 =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx1);
    ze_context_handle_t h2 =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx2);

    std::cout << "ctx1 ze_context handle: " << h1 << "\n";
    std::cout << "ctx2 ze_context handle: " << h2 << "\n";

    if (h1 != h2) {
      std::cerr << "FAIL: Two full-platform SYCL contexts have different "
                   "ze_context handles. They should both reuse the L0 "
                   "default context (URT-1145).\n";
      ++failed;
    } else {
      std::cout << "PASS: Both full-platform contexts share the same "
                   "ze_context handle.\n";
    }

    // Also verify the platform default context.
    sycl::context defCtx = platform.khr_get_default_context();
    ze_context_handle_t hDef =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(defCtx);
    std::cout << "platform default context handle: " << hDef << "\n";
    if (hDef != h1) {
      std::cerr << "FAIL: platform.khr_get_default_context() handle differs "
                   "from explicit full-platform context handle.\n";
      ++failed;
    } else {
      std::cout << "PASS: platform default context matches.\n";
    }
  }

  if (!anyTested) {
    std::cout << "No suitable L0 platform found, test skipped.\n";
    return 0;
  }

  return failed ? 1 : 0;
}
