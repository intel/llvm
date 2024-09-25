// REQUIRES: cpu
// RUN: %{build} -o %t.out
// RUN: env SYCL_UR_TRACE=2 %{run} %t.out 2>&1 | FileCheck %s

#include <sycl/detail/core.hpp>

#include <iostream>
#include <string>

int main() {
  const sycl::range<1> N{1};
  sycl::buffer<int, 1> Buf(
      N, {sycl::ext::oneapi::property::buffer::use_pinned_host_memory()});
  if (!Buf.has_property<
          sycl::ext::oneapi::property::buffer::use_pinned_host_memory>()) {
    std::cerr << "Buffer should have the use_pinned_host_memory property"
              << std::endl;
    return 1;
  }

  sycl::queue Q;
  Q.submit([&](sycl::handler &CGH) {
    auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);
    CGH.single_task<class init_a>([=]() {});
  });

  try {
    int Data = 0;
    sycl::buffer<int, 1> Buf(
        &Data, N,
        {sycl::ext::oneapi::property::buffer::use_pinned_host_memory()});
    // Expected that exception is thrown
    return 1;
  } catch (sycl::exception &E) {
    if (E.code() != sycl::errc::invalid ||
        std::string(E.what()).find(
            "The use_pinned_host_memory cannot be used with host pointer") ==
            std::string::npos) {
      return 1;
    }

    return 0;
  }
}

// CHECK:---> urMemBufferCreate
// CHECK-SAME: UR_MEM_FLAG_ALLOC_HOST_POINTER
