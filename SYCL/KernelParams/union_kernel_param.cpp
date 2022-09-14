// This test checks kernel execution with union type as kernel parameters.

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <cstdio>
#include <sycl/sycl.hpp>

union TestUnion {
public:
  int myint;
  char mychar;
  float myfloat;

  TestUnion() { myfloat = 0.0f; };
};

int main(int argc, char **argv) {
  TestUnion x;
  x.myfloat = 5.0f;
  float myfloat = 0.0f;

  sycl::queue queue;
  {
    sycl::buffer<float, 1> buf(&myfloat, 1);
    queue.submit([&](sycl::handler &cgh) {
      auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
      cgh.single_task<class test>([=]() { acc[0] = x.myfloat; });
    });
  }

  if (myfloat != 5.0f) {
    printf("FAILED\nmyfloat = %d\n", myfloat);
    return 1;
  }
  return 0;
}
