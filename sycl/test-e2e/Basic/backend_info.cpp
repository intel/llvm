// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//

//==--- backend_info.cpp - SYCL backend info test---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <sycl/detail/core.hpp>

using namespace sycl;

int main() {
  try {
    // Test get_backend_info for sycl::platform
    std::vector<platform> platform_list = platform::get_platforms();
    for (const auto &platform : platform_list) {
      std::cout << "  Backend device version: "
                << platform.get_backend_info<info::device::version>()
                << std::endl;
      std::cout << "  Backend platform version: "
                << platform.get_backend_info<info::platform::version>()
                << std::endl;
    }

    // Test get_backend_info for sycl::device
    std::vector<device> device_list =
        device::get_devices(info::device_type::gpu);
    for (const auto &device : device_list) {
      std::cout << "  Backend device version: "
                << device.get_backend_info<info::device::version>()
                << std::endl;
      std::cout << "  Backend platform version: "
                << device.get_backend_info<info::platform::version>()
                << std::endl;
    }

    // Test get_backend_info for sycl::queue
    queue q;
    std::cout << "  Backend device version: "
              << q.get_backend_info<info::device::version>() << std::endl;
    std::cout << "  Backend platform version: "
              << q.get_backend_info<info::platform::version>() << std::endl;

    // Test get_backend_info for sycl::context
    context Ctx = q.get_context();
    std::cout << "  Backend device version: "
              << Ctx.get_backend_info<info::device::version>() << std::endl;
    std::cout << "  Backend platform version: "
              << Ctx.get_backend_info<info::platform::version>() << std::endl;

    // Test get_backend_info for sycl::event
    event e = q.single_task([=]() { return; });
    std::cout << "  Backend device version: "
              << e.get_backend_info<info::device::version>() << std::endl;
    std::cout << "  Backend platform version: "
              << e.get_backend_info<info::platform::version>() << std::endl;

    // Test get_backend_info for sycl::kernel
    // Trivial kernel simply for testing
    buffer<int, 1> buf(range<1>(1));
    auto KernelID = sycl::get_kernel_id<class SingleTask>();
    auto KB = get_kernel_bundle<bundle_state::executable>(q.get_context(),
                                                          {KernelID});
    kernel krn = KB.get_kernel(KernelID);
    q.submit([&](handler &cgh) {
      auto acc = buf.get_access<access::mode::read_write>(cgh);
      cgh.single_task<class SingleTask>(krn, [=]() { acc[0] = acc[0] + 1; });
    });
    std::cout << "  Backend device version: "
              << krn.get_backend_info<info::device::version>() << std::endl;
    std::cout << "  Backend platform version: "
              << krn.get_backend_info<info::platform::version>() << std::endl;
  } catch (exception e) {
    // Check if the error code is the only allowed one: errc::backend_mismatch
    assert(e.code() == sycl::errc::backend_mismatch && "wrong error code");
    // If so, check if there're truly non-OpenCL backend(s) or it's an
    // unexpected error
    std::vector<platform> platform_list = platform::get_platforms();
    bool has_non_opencl_backend = false;
    for (const auto &platform : platform_list) {
      if (platform.get_backend() != backend::opencl) {
        has_non_opencl_backend = true;
        break;
      }
    }
    assert(has_non_opencl_backend && "unexpected error code");
  }
  std::cout << "  Backend info query tests passed" << std::endl;
  return 0;
}
