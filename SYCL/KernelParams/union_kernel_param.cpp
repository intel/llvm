// This test checks kernel execution with union type as kernel parameters.

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <cstdio>
#include <sycl/sycl.hpp>

union TestUnion {
public:
  int myint;
  char mychar;
  double mydouble;

  TestUnion() { mydouble = 0.0; };
};

int main(int argc, char **argv) {
  TestUnion x;
  x.mydouble = 5.0;
  double mydouble = 0.0;

  cl::sycl::queue queue;
  {
    cl::sycl::buffer<double, 1> buf(&mydouble, 1);
    queue.submit([&](cl::sycl::handler &cgh) {
      auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
      cgh.single_task<class test>([=]() { acc[0] = x.mydouble; });
    });
  }

  if (mydouble != 5.0) {
    printf("FAILED\nmydouble = %d\n", mydouble);
    return 1;
  }
  return 0;
}
