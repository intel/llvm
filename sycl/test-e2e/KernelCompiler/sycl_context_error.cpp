//==--- sycl_context_error.cpp --- kernel_compiler extension tests ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: (opencl || level_zero)

// UNSUPPORTED: accelerator
// UNSUPPORTED-INTENDED: while accelerator is AoT only, this cannot run there.

// RUN: %{build} -o %t.out
// RUN: %{run-unfiltered-devices} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/platform.hpp>
#include <sycl/usm.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

const std::string source = R"""(
    #include <sycl/sycl.hpp>
    namespace syclext = sycl::ext::oneapi;
    namespace syclexp = sycl::ext::oneapi::experimental;

    extern "C"
    SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
    void iota(float start, float *ptr) {
      size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
      ptr[id] = start + static_cast<float>(id);
    }
)""";

int main() {

  auto has_multiple_compatible_devices = [](sycl::platform platform) -> bool {
    auto devices = platform.get_devices();
    if (devices.size() < 2) {
      return false;
    }
    for (auto dev : devices) {
      if (!dev.ext_oneapi_can_build(syclexp::source_language::sycl)) {
        return false;
      }
    }
    return true;
  };

  std::vector<sycl::device> all_devices = [&]() -> std::vector<sycl::device> {
    for (auto platform : sycl::platform::get_platforms()) {
      if (has_multiple_compatible_devices(platform)) {
        return platform.get_devices();
      }
    }
    return {};
  }();

  if (all_devices.size() < 2) {
    std::cerr << "Cannot find platform with more than 1 device, skipping"
              << std::endl;
    return 0;
  }

  sycl::context single_device_context{all_devices.front()};

  // Create a source kernel bundle with a context that contains only one device.
  sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source> kb_src =
      syclexp::create_kernel_bundle_from_source(
          single_device_context, syclexp::source_language::sycl, source);

  // Compile the kernel.  There is no need to use the "registered_names"
  // property because the kernel is declared extern "C".
  try {
    syclexp::build(kb_src, all_devices);
    assert(false && "out-of-context device not detected");
  } catch (sycl::exception &e) {
    assert(e.code() == sycl::errc::invalid);
    assert(std::string(e.what()).find(
               "device not part of kernel_bundle context") !=
           std::string::npos);
  }
  return 0;
}
