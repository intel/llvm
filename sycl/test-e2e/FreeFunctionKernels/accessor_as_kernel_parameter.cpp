// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test verifies whether sycl::accessor can be used with free function
// kernels extension.

#include <sycl/ext/oneapi/free_function_queries.hpp>

#include "helpers.hpp"

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void globalScopeSingleFreeFunc(sycl::accessor<int, 1> Accessor,
                               size_t NumOfElements, int Value) {
  for (size_t I = 0; I < NumOfElements; ++I) {
    Accessor[I] = Value;
  }
}

namespace ns {
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<3>))
void nsNdRangeFreeFunc(sycl::accessor<int, 1> Accessor, int Value) {
  size_t Item =
      syclext::this_work_item::get_nd_item<3>().get_global_linear_id();
  Accessor[Item] = Value;
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<3>))
void nsNdRangeFreeFuncWith3DimAccessor(sycl::accessor<int, 3> Accessor,
                                       int Value) {
  sycl::nd_item<3> NdItem = syclext::this_work_item::get_nd_item<3>();
  Accessor[NdItem.get_global_id(0)][NdItem.get_global_id(1)]
          [NdItem.get_global_id(2)] = Value;
}

} // namespace ns

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void globalndRangeFreeFuncWith2DimAccessor(sycl::accessor<int, 2> Accessor,
                                           int Value) {
  sycl::nd_item<2> NdItem = syclext::this_work_item::get_nd_item<2>();
  Accessor[NdItem.get_group_linear_id()][NdItem.get_local_linear_id()] = Value;
}

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

  {
    // Check that sycl::accessor<2> is supported inside single_task free
    // function kernel.
    std::vector<int> ResultHostData(NumOfElements, 0);
    constexpr int ExpectedResultValue = 333;
    {
      sycl::range<2> BufRange{8, NumOfElements / 8};
      sycl::buffer<int, 2> Buffer(ResultHostData.data(), BufRange);
      sycl::kernel UsedKernel =
          getKernel<globalndRangeFreeFuncWith2DimAccessor>(Context);
      Queue.submit([&](sycl::handler &Handler) {
        sycl::accessor<int, 2> Accessor{Buffer, Handler};
        Handler.set_arg(0, Accessor);
        Handler.set_arg(1, ExpectedResultValue);
        sycl::nd_range<2> Ndr{{128, 8}, {16, 8}};
        Handler.parallel_for(Ndr, UsedKernel);
      });
    }
    Failed += performResultCheck(
        NumOfElements, ResultHostData.data(),
        "globalndRangeFreeFuncWith2DimAccessor with sycl::accessor<2>",
        ExpectedResultValue);
  }

  {
    // Check that sycl::accessor<3> is supported inside single_task free
    // function kernel.
    std::vector<int> ResultHostData(NumOfElements, 0);
    constexpr int ExpectedResultValue = 444;
    {
      sycl::range<3> BufRange{64, 4, 4};
      sycl::buffer<int, 3> Buffer(ResultHostData.data(), BufRange);
      sycl::kernel UsedKernel =
          getKernel<ns::nsNdRangeFreeFuncWith3DimAccessor>(Context);
      Queue.submit([&](sycl::handler &Handler) {
        sycl::accessor<int, 3> Accessor{Buffer, Handler};
        Handler.set_arg(0, Accessor);
        Handler.set_arg(1, ExpectedResultValue);
        sycl::nd_range<3> Ndr{{64, 4, 4}, {16, 4, 2}};
        Handler.parallel_for(Ndr, UsedKernel);
      });
    }
    Failed += performResultCheck(
        NumOfElements, ResultHostData.data(),
        "ns::nsNdRangeFreeFuncWith3DimAccessor with sycl::accessor<3>",
        ExpectedResultValue);
  }

  return Failed;
}
