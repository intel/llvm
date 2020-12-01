#include <CL/sycl.hpp>

int main() {
  cl::sycl::range<1> ndRng(1);
  int32_t kernelResult;
  cl::sycl::queue testQueue;
  {
    cl::sycl::buffer<int32_t, 1> buffer(&kernelResult, ndRng);
    testQueue.submit([&](cl::sycl::handler &h) {
      auto resultPtr =
          buffer.template get_access<cl::sycl::access::mode::write>(h);
      h.single_task<class kernel>([=]() {
        float inputData_0(0.1);
        float inputData_1(0.5);
        resultPtr[0] = cl::sycl::isordered(inputData_0, inputData_1);
      });
    });
  }
  // Should be 1 according to spec since it's a scalar type not a vector
  std::cout << "Kernel result: " << std::to_string(kernelResult) << std::endl;
  assert(kernelResult == 1 && "Incorrect result");

  return 0;
}