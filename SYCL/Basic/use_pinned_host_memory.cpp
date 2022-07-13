// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_PI_TRACE=2 %CPU_RUN_PLACEHOLDER %t.out 2>&1 %CPU_CHECK_PLACEHOLDER

#include <sycl/sycl.hpp>

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
  } catch (sycl::invalid_object_error &E) {
    if (std::string(E.what()).find(
            "The use_pinned_host_memory cannot be used with host pointer") ==
        std::string::npos) {
      return 1;
    }

    return 0;
  }
}

// CHECK:---> piMemBufferCreate
// CHECK-NEXT: {{.*}} : {{.*}}
// CHECK-NEXT: {{.*}} : 17
