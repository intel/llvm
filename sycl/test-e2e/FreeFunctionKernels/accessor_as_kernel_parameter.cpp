// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test verifies whether sycl::accessor can be used with free function
// kernels extension.

#include <sycl/ext/oneapi/free_function_queries.hpp>

#include "helpers.hpp"

template <int Dims>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void globalScopeSingleFreeFunc(
    sycl::accessor<int, Dims, sycl::access::mode::read_write,
                   sycl::access::target::device,
                   sycl::access::placeholder::false_t>
        Accessor,
    int Value) {
  for (auto &Elem : Accessor)
    Elem = Value;
}
namespace ns {
template <int Dims>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<Dims>))
void nsNdRangeFreeFunc(sycl::accessor<int, Dims, sycl::access::mode::read_write,
                                      sycl::access::target::device,
                                      sycl::access::placeholder::false_t>
                           Accessor,
                       int Value) {
  auto Item = syclext::this_work_item::get_nd_item<Dims>().get_global_id();
  Accessor[Item] = Value;
}
} // namespace ns

template <int Dims>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<Dims>))
void ndRangeFreeFuncMultipleParameters(
    sycl::accessor<int, Dims, sycl::access::mode::read,
                   sycl::access::target::device,
                   sycl::access::placeholder::false_t>
        InputAAcc,
    sycl::accessor<int, Dims, sycl::access::mode::read,
                   sycl::access::target::device,
                   sycl::access::placeholder::false_t>
        InputBAcc,
    sycl::accessor<int, Dims, sycl::access::mode::write,
                   sycl::access::target::device,
                   sycl::access::placeholder::false_t>
        ResultAcc) {
  auto Item = syclext::this_work_item::get_nd_item<Dims>().get_global_id();
  ResultAcc[Item] = InputAAcc[Item] + InputBAcc[Item];
}

// TODO: Need to add checks for a static member functions of a class as free
// function kernel.

template <auto Func, size_t Dims>
int runSingleTaskTest(sycl::queue &Queue, sycl::context &Context,
                      sycl::range<Dims> NumOfElementsPerDim,
                      std::string_view ErrorMessage,
                      const int ExpectedResultValue) {
  sycl::kernel UsedKernel = getKernel<Func>(Context);
  std::vector<int> ResultData(NumOfElementsPerDim.size(), 0);
  {
    sycl::buffer<int, Dims> Buffer(ResultData.data(), NumOfElementsPerDim);
    Queue.submit([&](sycl::handler &Handler) {
      sycl::accessor<int, Dims> Accessor{Buffer, Handler};
      Handler.set_args(Accessor, ExpectedResultValue);
      Handler.single_task(UsedKernel);
    });
  }
  return performResultCheck(NumOfElementsPerDim.size(), ResultData.data(),
                            ErrorMessage, ExpectedResultValue);
}
template <auto Func, size_t Dims>
int runNdRangeTest(sycl::queue &Queue, sycl::context &Context,
                   sycl::nd_range<Dims> NdRange, std::string_view ErrorMessage,
                   const int ExpectedResultValue) {
  sycl::kernel UsedKernel = getKernel<Func>(Context);
  std::vector<int> ResultData(NdRange.get_global_range().size(), 0);
  {
    sycl::buffer<int, Dims> Buffer(ResultData.data(),
                                   NdRange.get_global_range());
    Queue.submit([&](sycl::handler &Handler) {
      sycl::accessor<int, Dims> Accessor{Buffer, Handler};
      Handler.set_args(Accessor, ExpectedResultValue);
      Handler.parallel_for(NdRange, UsedKernel);
    });
  }
  return performResultCheck(NdRange.get_global_range().size(),
                            ResultData.data(), ErrorMessage,
                            ExpectedResultValue);
}

template <auto Func, size_t Dims>
int runNdRangeTestMultipleParameters(sycl::queue &Queue, sycl::context &Context,
                                     sycl::nd_range<Dims> NdRange,
                                     std::string_view ErrorMessage,
                                     sycl::range<3> Values) {
  sycl::kernel UsedKernel = getKernel<Func>(Context);
  std::vector<int> InputAData(NdRange.get_global_range().size(), Values[0]);
  std::vector<int> InputBData(NdRange.get_global_range().size(), Values[1]);
  std::vector<int> ResultData(NdRange.get_global_range().size(), 0);

  {
    sycl::buffer<int, Dims> InputABuffer(InputAData.data(),
                                         NdRange.get_global_range());
    sycl::buffer<int, Dims> InputBBuffer(InputBData.data(),
                                         NdRange.get_global_range());
    sycl::buffer<int, Dims> ResultBuffer(ResultData.data(),
                                         NdRange.get_global_range());
    Queue.submit([&](sycl::handler &Handler) {
      sycl::accessor<int, Dims, sycl::access::mode::read,
                     sycl::access::target::device>
          InputAAcc{InputABuffer, Handler};
      sycl::accessor<int, Dims, sycl::access::mode::read,
                     sycl::access::target::device>
          InputBAcc{InputBBuffer, Handler};
      sycl::accessor<int, Dims, sycl::access::mode::write> ResultAcc{
          ResultBuffer, Handler};
      Handler.set_args(InputAAcc, InputBAcc, ResultAcc);
      Handler.parallel_for(NdRange, UsedKernel);
    });
  }
  return performResultCheck(NdRange.get_global_range().size(),
                            ResultData.data(), ErrorMessage, Values[2]);
}
int main() {

  int Failed = 0;
  sycl::queue Queue;
  sycl::context Context = Queue.get_context();
  {
    // Check that sycl::accessor is supported inside single_task free function
    // kernel
    Failed += runSingleTaskTest<globalScopeSingleFreeFunc<1>, 1>(
        Queue, Context, sycl::range<1>{10},
        "globalScopeSingleFreeFunc with sycl::accessor<1>", 1);
    Failed += runSingleTaskTest<globalScopeSingleFreeFunc<2>, 2>(
        Queue, Context, sycl::range<2>{10, 10},
        "globalScopeSingleFreeFunc with sycl::accessor<2>", 2);
    Failed += runSingleTaskTest<globalScopeSingleFreeFunc<3>, 3>(
        Queue, Context, sycl::range<3>{5, 5, 5},
        "globalScopeSingleFreeFunc with sycl::accessor<3>", 3);
  }

  {
    // Check that sycl::accessor is supported inside nd_range free function
    // kernel
    Failed += runNdRangeTest<ns::nsNdRangeFreeFunc<1>, 1>(
        Queue, Context, sycl::nd_range{sycl::range{10}, sycl::range{2}},
        "ns::nsNdRangeFreeFunc with sycl::accessor<1>", 4);
    Failed += runNdRangeTest<ns::nsNdRangeFreeFunc<2>, 2>(
        Queue, Context, sycl::nd_range{sycl::range{16, 16}, sycl::range{4, 4}},
        "ns::nsNdRangeFreeFunc with sycl::accessor<2>", 5);
    Failed += runNdRangeTest<ns::nsNdRangeFreeFunc<3>, 3>(
        Queue, Context,
        sycl::nd_range{sycl::range{10, 10, 10}, sycl::range{2, 2, 2}},
        "ns::nsNdRangeFreeFunc with sycl::accessor<3>", 6);
  }

  {
    // Check that multiple sycl::accessor are supported inside nd_range free
    // function kernel
    Failed +=
        runNdRangeTestMultipleParameters<ndRangeFreeFuncMultipleParameters<1>,
                                         1>(
            Queue, Context, sycl::nd_range{sycl::range{10}, sycl::range{2}},
            "ndRangeFreeFuncMultipleParameters with multiple sycl::accessor<1>",
            sycl::range{111, 111, 222});
    Failed +=
        runNdRangeTestMultipleParameters<ndRangeFreeFuncMultipleParameters<2>,
                                         2>(
            Queue, Context,
            sycl::nd_range{sycl::range{16, 16}, sycl::range{4, 4}},
            "ndRangeFreeFuncMultipleParameters with multiple sycl::accessor<2>",
            sycl::range{222, 222, 444});
    Failed +=
        runNdRangeTestMultipleParameters<ndRangeFreeFuncMultipleParameters<3>,
                                         3>(
            Queue, Context,
            sycl::nd_range{sycl::range{10, 10, 10}, sycl::range{2, 2, 2}},
            "ndRangeFreeFuncMultipleParameters with multiple sycl::accessor<3>",
            sycl::range{444, 444, 888});
  }
  return Failed;
}
