// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsycl-unnamed-lambda %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
#include <sycl/sycl.hpp>

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
  if (!D.has(sycl::aspect::fp16))
    return 0; // Skip the test if halfs are not supported

  sycl::buffer<sycl::half> Buf(1);

  Q.submit([&](sycl::handler &CGH) {
    auto Acc = Buf.get_access<sycl::access::mode::write>(CGH);
    CGH.single_task([=]() { Acc[0] = 1; });
  });

  Q.wait_and_throw();

  auto Acc = Buf.get_access<sycl::access::mode::read>();
  if (1 != Acc[0]) {
    std::cerr << "Incorrect result, got: " << Acc[0] << ", expected: 1"
              << std::endl;
    return 1;
  }

  return 0;
}
