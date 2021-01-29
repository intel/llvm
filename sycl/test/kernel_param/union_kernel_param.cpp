// This test checks kernel execution with union type as kernel parameters.

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %RUN_ON_HOST %t.out

#include <CL/sycl.hpp>
#include <cstdio>

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
