// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test verifies whether a range<Dimensions> can be passed as a kernel
// parameter to a free function kernel.

#include <sycl/usm.hpp>

#include <sycl/ext/oneapi/free_function_queries.hpp>

#include "helpers.hpp"

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void globalScopeSingleFreeFuncKernel(int *Ptr, size_t NumOfElements,
                                     sycl::range<1> Range) {
  for (size_t i = 0; i < NumOfElements; ++i) {
    Ptr[i] = static_cast<int>(Range.size());
  }
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void globalScopeNdRangeFreeFuncKernel(int *Ptr, sycl::range<2> Range) {
  size_t Item =
      syclext::this_work_item::get_nd_item<2>().get_global_linear_id();
  Ptr[Item] = static_cast<int>(Range.size());
}

namespace ns {
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void nsSingleFreeFuncKernel(int *Ptr, size_t NumOfElements,
                            sycl::range<1> Range) {
  for (size_t i = 0; i < NumOfElements; ++i) {
    Ptr[i] = static_cast<int>(Range.size());
  }
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<3>))
void nsNdRangeFreeFuncKernel(int *Ptr, sycl::range<3> Range) {
  size_t Item =
      syclext::this_work_item::get_nd_item<3>().get_global_linear_id();
  Ptr[Item] = static_cast<int>(Range.size());
}
} // namespace ns

// TODO: Need to add checks for a static member functions of a class as free
// function kerenl

int main() {

  int Failed = 0;
  sycl::queue Queue;
  sycl::context Context = Queue.get_context();
  constexpr size_t NumOfElements = 1024;
  int *Data = sycl::malloc_shared<int>(NumOfElements, Queue);

  {
    std::fill(Data, Data + NumOfElements, 0);
    sycl::kernel UsedKernel = getKernel<ns::nsSingleFreeFuncKernel>(Context);

    sycl::range<1> Range{11};
    int ExpectedResultValue = static_cast<int>(Range.size());
    Queue
        .submit([&](sycl::handler &Handler) {
          Handler.set_args(Data, NumOfElements, Range);
          Handler.single_task(UsedKernel);
        })
        .wait();

    Failed += performResultCheck(
        NumOfElements, Data, "ns::nsSingleFreeFuncKernel", ExpectedResultValue);
  }

  {
    std::fill(Data, Data + NumOfElements, 0);
    sycl::kernel UsedKernel = getKernel<ns::nsNdRangeFreeFuncKernel>(Context);
    sycl::range<3> Range{22, 22, 22};
    int ExpectedResultValue = static_cast<int>(Range.size());
    Queue
        .submit([&](sycl::handler &Handler) {
          Handler.set_arg(0, Data);
          Handler.set_arg(1, Range);
          sycl::nd_range<3> Ndr{{4, 4, NumOfElements / 16}, {4, 4, 4}};
          Handler.parallel_for(Ndr, UsedKernel);
        })
        .wait();

    Failed +=
        performResultCheck(NumOfElements, Data, "ns::nsNdRangeFreeFuncKernel",
                           ExpectedResultValue);
  }

  {
    std::fill(Data, Data + NumOfElements, 0);
    sycl::kernel UsedKernel =
        getKernel<globalScopeSingleFreeFuncKernel>(Context);
    sycl::range<1> Range{33};
    int ExpectedResultValue = static_cast<int>(Range.size());
    Queue
        .submit([&](sycl::handler &Handler) {
          Handler.set_arg(0, Data);
          Handler.set_arg(1, NumOfElements);
          Handler.set_arg(2, Range);
          Handler.single_task(UsedKernel);
        })
        .wait();

    Failed += performResultCheck(NumOfElements, Data,
                                 "globalScopeSingleFreeFuncKernel",
                                 ExpectedResultValue);
  }

  {
    std::fill(Data, Data + NumOfElements, 0);
    sycl::kernel UsedKernel =
        getKernel<globalScopeNdRangeFreeFuncKernel>(Context);
    sycl::range<2> Range{44, 44};
    int ExpectedResultValue = static_cast<int>(Range.size());
    Queue
        .submit([&](sycl::handler &Handler) {
          Handler.set_args(Data, Range);
          sycl::nd_range<2> Ndr{{8, NumOfElements / 8}, {8, 8}};
          Handler.parallel_for(Ndr, UsedKernel);
        })
        .wait();

    Failed += performResultCheck(NumOfElements, Data,
                                 "globalScopeNdRangeFreeFuncKernel",
                                 ExpectedResultValue);
  }

  sycl::free(Data, Queue);
  return Failed;
}
