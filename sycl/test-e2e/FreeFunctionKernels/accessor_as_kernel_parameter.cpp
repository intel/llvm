// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test verifies whether sycl::accessor can be used with free function
// kernels extension.

#include <sycl/ext/oneapi/free_function_queries.hpp>

#include "helpers.hpp"

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void globalScopeSingleFreeFunc(sycl::accessor<int, 1> Accessor,
                               size_t NumOfElements, int Value) {
  for (size_t i = 0; i < NumOfElements; ++i) {
    Accessor[i] = Value;
  }
}

namespace ns {
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<3>))
void nsNdRangeFreeFunc(sycl::accessor<int, 1> Accessor, int Value) {
  size_t Item =
      syclext::this_work_item::get_nd_item<3>().get_global_linear_id();
  Accessor[Item] = Value;
}
} // namespace ns

// TODO: Need to add checks for a static member functions of a class as free
// function kerenl

int main() {

  int Failed = 0;
  sycl::queue Queue;
  sycl::context Context = Queue.get_context();
  constexpr size_t NumOfElements = 1024;
  {
    // Check that sycl::accessor is supported inside nd_range free function
    // kernel.
    std::vector<int> ResultHostData(NumOfElements, 0);
    constexpr int ExpectedResultValue = 111;
    {
      sycl::buffer<int, 1> Buffer(ResultHostData);
      sycl::kernel UsedKernel = getKernel<ns::nsNdRangeFreeFunc>(Context);
      Queue.submit([&](sycl::handler &Handler) {
        sycl::accessor<int, 1> Accessor{Buffer, Handler};
        Handler.set_args(Accessor, ExpectedResultValue);
        sycl::nd_range<3> Ndr{{4, 4, NumOfElements / 16}, {4, 4, 4}};
        Handler.parallel_for(Ndr, UsedKernel);
      });
    }

    Failed += performResultCheck(NumOfElements, ResultHostData.data(),
                                 "ns::nsNdRangeFreeFunc with sycl::accessor",
                                 ExpectedResultValue);
  }

  {
    // Check that sycl::accessor is supported inside single_task free function
    // kernel.
    std::vector<int> ResultHostData(NumOfElements, 0);
    constexpr int ExpectedResultValue = 222;
    {
      sycl::buffer<int, 1> Buffer(ResultHostData);
      sycl::kernel UsedKernel = getKernel<globalScopeSingleFreeFunc>(Context);
      Queue.submit([&](sycl::handler &Handler) {
        sycl::accessor<int, 1> Accessor{Buffer, Handler};
        Handler.set_arg(0, Accessor);
        Handler.set_arg(1, NumOfElements);
        Handler.set_arg(2, ExpectedResultValue);
        Handler.single_task(UsedKernel);
      });
    }
    Failed += performResultCheck(
        NumOfElements, ResultHostData.data(),
        "globalScopeSingleFreeFunc with sycl::accessor", ExpectedResultValue);
  }

  return Failed;
}
