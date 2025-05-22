// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test verifies whether USM can be used with free function kernels
// extension.

#include <sycl/usm.hpp>

#include <sycl/ext/oneapi/free_function_queries.hpp>

#include "helpers.hpp"

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void globalScopeSingleFreeFunc(int *Ptr, size_t NumOfElements, int Value) {
  for (size_t i = 0; i < NumOfElements; ++i) {
    Ptr[i] = Value;
  }
}

namespace ns {
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<3>))
void nsNdRangeFreeFunc(int *Ptr, int Value) {
  size_t Item =
      syclext::this_work_item::get_nd_item<3>().get_global_linear_id();
  Ptr[Item] = Value;
}
} // namespace ns

// TODO: Need to add checks for a static member functions of a class as free
// function kerenl

int main() {

  int Failed = 0;
  sycl::queue Queue;
  sycl::context Context = Queue.get_context();
  constexpr size_t NumOfElements = 1024;
  // Check that device type of USM allocation is supported inside free function
  // kernel.
  {
    int *HostDataPtr = new int[NumOfElements];
    constexpr int ExpectedResultValue = 111;
    int *DeviceDataPtr = sycl::malloc_device<int>(NumOfElements, Queue);
    sycl::kernel UsedKernel = getKernel<ns::nsNdRangeFreeFunc>(Context);
    Queue
        .submit([&](sycl::handler &Handler) {
          Handler.set_args(DeviceDataPtr, ExpectedResultValue);
          sycl::nd_range<3> Ndr{{4, 4, NumOfElements / 16}, {4, 4, 4}};
          Handler.parallel_for(Ndr, UsedKernel);
        })
        .wait();

    Queue.copy(DeviceDataPtr, HostDataPtr, NumOfElements).wait();

    Failed +=
        performResultCheck(NumOfElements, HostDataPtr,
                           "ns::nsNdRangeFreeFunc with USM device allocation",
                           ExpectedResultValue);
    sycl::free(DeviceDataPtr, Queue);
  }

  // Check that host type of USM allocation is supported inside free function
  // kernel.
  {
    constexpr int ExpectedResultValue = 222;
    int *DataPtr = sycl::malloc_host<int>(NumOfElements, Queue);
    sycl::kernel UsedKernel = getKernel<globalScopeSingleFreeFunc>(Context);
    Queue
        .submit([&](sycl::handler &Handler) {
          Handler.set_arg(0, DataPtr);
          Handler.set_arg(1, NumOfElements);
          Handler.set_arg(2, ExpectedResultValue);
          Handler.single_task(UsedKernel);
        })
        .wait();
    Failed +=
        performResultCheck(NumOfElements, DataPtr,
                           "globalScopeSingleFreeFunc with USM host allocation",
                           ExpectedResultValue);
    sycl::free(DataPtr, Queue);
  }

  // Check that shared type of USM allocation is supported inside free function
  // kernel.

  {
    constexpr int ExpectedResultValue = 333;
    int *DataPtr = sycl::malloc_shared<int>(NumOfElements, Queue);
    sycl::kernel UsedKernel = getKernel<ns::nsNdRangeFreeFunc>(Context);
    Queue
        .submit([&](sycl::handler &Handler) {
          Handler.set_args(DataPtr, ExpectedResultValue);
          sycl::nd_range<3> Ndr{{4, 4, NumOfElements / 16}, {4, 4, 4}};
          Handler.parallel_for(Ndr, UsedKernel);
        })
        .wait();

    Failed +=
        performResultCheck(NumOfElements, DataPtr,
                           "ns::nsNdRangeFreeFunc with USM device allocation",
                           ExpectedResultValue);
    sycl::free(DataPtr, Queue);
  }

  return Failed;
}
