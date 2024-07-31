// This test checks kernel execution with union type as kernel parameters.

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <cstdio>
#include <sycl/detail/core.hpp>

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
    std::cout << "FAILED\nmyfloat = " << myfloat << std::endl;
    return 1;
  }
  return 0;
}
