// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUNx: %CPU_RUN_PLACEHOLDER %t.out
// RUNx: %GPU_RUN_PLACEHOLDER %t.out
// RUNx: %ACC_RUN_PLACEHOLDER %t.out

#include <sycl/sycl.hpp>

int main() {
  sycl::range<1> ndRng(3);
  int32_t kernelResult[3];
  sycl::queue testQueue;
  {
    sycl::buffer<int32_t, 1> buffer(&kernelResult[0], ndRng);
    testQueue.submit([&](sycl::handler &h) {
      auto resultPtr = buffer.template get_access<sycl::access::mode::write>(h);
      h.single_task<class kernel>([=]() {
        float inputData_0F(0.1);
        float inputData_1F(0.5);
        resultPtr[0] = sycl::isordered(inputData_0F, inputData_1F);

        double inputData_0D(0.2);
        double inputData_1D(0.3);
        resultPtr[1] = sycl::isordered(inputData_0D, inputData_1D);

        sycl::half inputData_0H(0.3);
        sycl::half inputData_1H(0.9);
        resultPtr[2] = sycl::isordered(inputData_0H, inputData_1H);
      });
    });
  }
  // Should be 1 according to spec since it's a scalar type not a vector
  assert(kernelResult[0] == 1 && "Incorrect result");
  assert(kernelResult[1] == 1 && "Incorrect result");
  assert(kernelResult[2] == 1 && "Incorrect result");

  return 0;
}
