// REQUIRES: aspect-fp16
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
#include <sycl/detail/core.hpp>

#include <iostream>

int main() {
  auto AsyncHandler = [](sycl::exception_list EL) {
    for (std::exception_ptr const &P : EL) {
      try {
        std::rethrow_exception(P);
      } catch (std::exception const &E) {
        std::cerr << "Caught async SYCL exception: " << E.what() << std::endl;
      }
    }
  };

  sycl::queue Q(AsyncHandler);

  sycl::device D = Q.get_device();

  sycl::buffer<sycl::half> Buf(1);

  Q.submit([&](sycl::handler &CGH) {
    auto Acc = Buf.get_access<sycl::access::mode::write>(CGH);
    CGH.single_task([=]() { Acc[0] = 1; });
  });

  Q.wait_and_throw();

  auto Acc = Buf.get_host_access();
  if (1 != Acc[0]) {
    std::cerr << "Incorrect result, got: " << Acc[0] << ", expected: 1"
              << std::endl;
    return 1;
  }

  return 0;
}
