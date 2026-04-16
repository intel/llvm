// REQUIRES: level_zero, level_zero_dev_kit
// REQUIRES: level_zero_v2_adapter
// RUN: %{build} %level_zero_options -o %t.out
// RUN: %{run} %t.out

// Test that the SYCL platform default context reuses the Level Zero driver
// default context (zeDriverGetDefaultContext) when all root devices of a
// platform are requested.  Verifies URT-1145.
//
// The test requires the driver to support zeDriverGetDefaultContext.
// If the extension is absent the test exits with a skip-style success because
// the feature is driver-gated and there is nothing to verify.
//
// Before the fix: urContextCreate always calls zeContextCreate, so two
// full-platform contexts have different ze_context handles -> test FAILS.
// After the fix:  urContextCreate reuses the driver default context, so both
// calls return the same handle -> test PASSES.

#include <dlfcn.h>
#include <iostream>
#include <level_zero/ze_api.h>
#include <sycl/backend.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/platform.hpp>
#include <vector>

// zeDriverGetDefaultContext is a core Level Zero v1.15 API.
// Resolve it at runtime via dlsym to avoid a hard link-time dependency on a
// specific version of libze_loader.
typedef ze_context_handle_t (*pfnZeDriverGetDefaultContext_t)(
    ze_driver_handle_t);

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

    // Secondary check: compare against zeDriverGetDefaultContext directly
    // (core L0 v1.15 API, returns the handle or nullptr on failure).
    // Use dlsym to avoid a hard link-time dependency on a specific ze_loader.
    ze_driver_handle_t zeDriver =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(platform);
    auto *pfnGetDefaultCtx = reinterpret_cast<pfnZeDriverGetDefaultContext_t>(
        dlsym(RTLD_DEFAULT, "zeDriverGetDefaultContext"));
    if (pfnGetDefaultCtx) {
      ze_context_handle_t zeDefaultCtx = pfnGetDefaultCtx(zeDriver);
      if (zeDefaultCtx != nullptr) {
        std::cout << "zeDriverGetDefaultContext handle: " << zeDefaultCtx
                  << "\n";
        if (h1 != zeDefaultCtx) {
          std::cerr << "FAIL: Full-platform context handle does not match "
                       "zeDriverGetDefaultContext.\n";
          ++failed;
        } else {
          std::cout << "PASS: Matches zeDriverGetDefaultContext.\n";
        }
      } else {
        std::cout << "zeDriverGetDefaultContext returned nullptr "
                     "(secondary check skipped).\n";
      }
    } else {
      std::cout << "zeDriverGetDefaultContext not found at runtime "
                   "(secondary check skipped).\n";
    }
  }

  if (!anyTested) {
    std::cout << "No suitable L0 platform found, test skipped.\n";
    return 0;
  }

  return failed ? 1 : 0;
}
