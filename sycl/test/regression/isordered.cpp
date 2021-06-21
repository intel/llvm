// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env SYCL_DEVICE_FILTER=host %t.out

#include <CL/sycl.hpp>

int main() {
  cl::sycl::range<1> ndRng(3);
  int32_t kernelResult[3];
  cl::sycl::queue testQueue;
  {
    cl::sycl::buffer<int32_t, 1> buffer(&kernelResult[0], ndRng);
    testQueue.submit([&](cl::sycl::handler &h) {
      auto resultPtr =
          buffer.template get_access<cl::sycl::access::mode::write>(h);
      h.single_task<class kernel>([=]() {
        float inputData_0F(0.1);
        float inputData_1F(0.5);
        resultPtr[0] = cl::sycl::isordered(inputData_0F, inputData_1F);

        double inputData_0D(0.2);
        double inputData_1D(0.3);
        resultPtr[1] = cl::sycl::isordered(inputData_0D, inputData_1D);

        half inputData_0H(0.3);
        half inputData_1H(0.9);
        resultPtr[2] = cl::sycl::isordered(inputData_0H, inputData_1H);
      });
    });
  }
  // Should be 1 according to spec since it's a scalar type not a vector
  assert(kernelResult[0] == 1 && "Incorrect result");
  assert(kernelResult[1] == 1 && "Incorrect result");
  assert(kernelResult[2] == 1 && "Incorrect result");

  return 0;
}
