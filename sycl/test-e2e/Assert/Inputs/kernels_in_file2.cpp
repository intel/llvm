#include "kernels_in_file2.hpp"

#ifdef DEFINE_NDEBUG_INFILE2
#define NDEBUG
#else
#undef NDEBUG
#endif

#include <cassert>

using namespace sycl;
using namespace sycl::access;

int calculus(int X) {
  assert(X && "this message from calculus");
  return X * 2;
}

void check_nil(int value) { assert(value && "this message from file2"); }

static constexpr size_t BUFFER_SIZE = 4;

void enqueueKernel_1_fromFile2(queue *Q) {
  sycl::range<1> numOfItems{BUFFER_SIZE};
  sycl::buffer<int, 1> Buf(numOfItems);

  Q->submit([&](handler &CGH) {
    auto Acc = Buf.template get_access<mode::read_write>(CGH);

    CGH.parallel_for<class kernel1_from_separate_file>(
        numOfItems, [=](sycl::id<1> wiID) { check_nil(Acc[wiID]); });
  });
}

void enqueueKernel_2_fromFile2(queue *Q) {
  sycl::range<1> numOfItems{BUFFER_SIZE};
  sycl::buffer<int, 1> Buf(numOfItems);

  Q->submit([&](handler &CGH) {
    auto Acc = Buf.template get_access<mode::read_write>(CGH);

    CGH.parallel_for<class kernel2_from_separate_file>(
        numOfItems, [=](sycl::id<1> wiID) { check_nil(Acc[wiID]); });
  });
}
