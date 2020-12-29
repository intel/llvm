// RUN: %clangxx -fsycl %s -o %t.out

#include <CL/sycl.hpp>

int main() {
  cl::sycl::vec<int8_t, 2> inputVec = cl::sycl::vec<int8_t, 2>(0, 1);
 
  auto asVec =
      inputVec.template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1>()
          .template as<cl::sycl::vec<int16_t, 1>>();

  return 0;