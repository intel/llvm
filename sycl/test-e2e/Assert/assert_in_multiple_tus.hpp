#include "Inputs/kernels_in_file2.hpp"
#include <iostream>
#include <sycl/detail/core.hpp>

#include <sycl/builtins.hpp>

#ifdef DEFINE_NDEBUG_INFILE1
#define NDEBUG
#else
#undef NDEBUG
#endif

#include <cassert>

using namespace sycl;
using namespace sycl::access;

static constexpr size_t BUFFER_SIZE = 16;

int checkFunction() {
  int X = calculus(0);
  assert(X && "Nil in result");
  return X;
}

void enqueueKernel_1_fromFile1(queue *Q) {
  sycl::range<1> numOfItems{BUFFER_SIZE};
  sycl::buffer<int, 1> Buf(numOfItems);

  Q->submit([&](handler &CGH) {
    auto Acc = Buf.template get_access<mode::read_write>(CGH);

    CGH.parallel_for<class Kernel_1>(
        sycl::nd_range(Buf.get_range(), sycl::range<1>(4)),
        [=](sycl::id<1> wiID) {
          int X = 0;
          if (wiID == 5)
            X = checkFunction();
          Acc[wiID] = X;
        });
  });
}

int main(int Argc, const char *Argv[]) {

  queue Q;
  enqueueKernel_1_fromFile1(&Q);
  enqueueKernel_2_fromFile2(&Q);
  Q.wait();

  std::cout << "The test ended." << std::endl;
  return 0;
}
