// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test verifies that we can use scoped enum types as arguments in free
// function kernels.

#include "helpers.hpp"
#include <cassert>
#include <sycl/ext/oneapi/free_function_queries.hpp>

using namespace sycl;
using rAccType = sycl::accessor<int, 1, sycl::access::mode::read>;
using wAccType = sycl::accessor<int, 1, sycl::access::mode::write>;
using rwAccType = sycl::accessor<int, 1, sycl::access::mode::read_write>;

using flagType = sycl::accessor<bool, 1>;

template <typename T, int Dims, access::mode modeT>
bool hasAccessorMode(sycl::accessor<T, Dims, modeT> acc, access::mode mode) {
  return modeT == mode;
}

template <typename AccType>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void verifyAccessorMode(AccType acc, access::mode accMode, flagType flagAcc) {
  flagAcc[0] = hasAccessorMode(acc, accMode);
}

int main() {
  sycl::queue Queue;
  sycl::context Context = Queue.get_context();
  sycl::kernel Kernel1 = getKernel<verifyAccessorMode<rAccType>>(Context);
  sycl::kernel Kernel2 = getKernel<verifyAccessorMode<wAccType>>(Context);
  sycl::kernel Kernel3 = getKernel<verifyAccessorMode<rwAccType>>(Context);

  const auto rMode = sycl::access::mode::read;
  const auto wMode = sycl::access::mode::write;
  const auto rwMode = sycl::access::mode::read_write;

  bool flag1, flag2, flag3;
  flag1 = (flag2 = (flag3 = false));
  {
    sycl::buffer<bool, 1> flagBuffer1(&flag1, 1);
    sycl::buffer<bool, 1> flagBuffer2(&flag2, 1);
    sycl::buffer<bool, 1> flagBuffer3(&flag3, 1);
    Queue.submit([&](sycl::handler &Handler) {
      rAccType acc1;
      flagType flagAcc1{flagBuffer1, Handler};
      Handler.set_args(acc1, rMode, flagAcc1);
      Handler.single_task(Kernel1);
    });

    Queue.submit([&](sycl::handler &Handler) {
      wAccType acc2;
      flagType flagAcc2{flagBuffer2, Handler};
      Handler.set_args(acc2, wMode, flagAcc2);
      Handler.single_task(Kernel2);
    });

    Queue.submit([&](sycl::handler &Handler) {
      rwAccType acc3;
      flagType flagAcc3{flagBuffer3, Handler};
      Handler.set_args(acc3, rwMode, flagAcc3);
      Handler.single_task(Kernel3);
    });
  }
  assert(flag1 && flag2 && flag3);
}
