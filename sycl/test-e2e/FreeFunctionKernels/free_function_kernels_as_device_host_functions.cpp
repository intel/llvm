// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test verifies whether free function kernel can be used as device
// function within another kernel or can be used as normal host function.

#include <numeric>

#include <sycl/usm.hpp>

#include <sycl/ext/oneapi/free_function_queries.hpp>

#include "helpers.hpp"

template <typename T, int Dims>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<Dims>))
void setValues(T *DataPtr, size_t N, T ExpectedResult) {
#if __SYCL_DEVICE_ONLY__
  auto GlobalLinId =
      syclext::this_work_item::get_nd_item<Dims>().get_global_linear_id();
  DataPtr[GlobalLinId] = ExpectedResult;
#else
  for (size_t I = 0; I < N; ++I)
    DataPtr[I] = ExpectedResult;
#endif
}

template <typename T>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void performReverse(T *DataPtr, size_t N) {
  for (size_t I = 0, J = N - 1; I < J; ++I, --J) {
    T Temp = DataPtr[I];
    DataPtr[I] = DataPtr[J];
    DataPtr[J] = Temp;
  }
}

namespace ns {
template <typename T>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void singleTaskKernel(T *DataPtr, size_t N) {
  performReverse(DataPtr, N);
}
} // namespace ns

template <typename T, int Dims>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<Dims>))
void ndRangekernel(T *DataPtr, size_t N, T ExpectedResult) {
  setValues<T, Dims>(DataPtr, N, ExpectedResult);
}

template <auto Func, size_t Dims>
int runNdRangeKernel(sycl::queue &Queue, sycl::context &Context,
                     sycl::nd_range<Dims> NdRange,
                     const int ExpectedResultValue, std::string_view TestName) {
  sycl::kernel UsedKernel = getKernel<Func>(Context);
  const int NumberOfElements = NdRange.get_global_range().size();
  int *DataPtr = sycl::malloc_shared<int>(NumberOfElements, Queue);
  std::fill(DataPtr, DataPtr + NumberOfElements, 0);
  Queue
      .submit([&](sycl::handler &Handler) {
        Handler.set_args(DataPtr, NumberOfElements, ExpectedResultValue);
        Handler.parallel_for(NdRange, UsedKernel);
      })
      .wait();

  int Failed = performResultCheck(NumberOfElements, DataPtr, TestName,
                                  ExpectedResultValue);
  sycl::free(DataPtr, Queue);
  return Failed;
}

template <auto Func, typename T, size_t NumOfElements>
int runSingleTaskKernel(sycl::queue &Queue, sycl::context &Context,
                        std::string_view TestName) {
  sycl::kernel UsedKernel = getKernel<Func>(Context);
  std::array<T, NumOfElements> ExpectedResultValues;
  std::iota(ExpectedResultValues.begin(), ExpectedResultValues.end(), 0);
  std::reverse(ExpectedResultValues.begin(), ExpectedResultValues.end());

  T *DataPtr = sycl::malloc_shared<T>(NumOfElements, Queue);
  std::iota(DataPtr, DataPtr + NumOfElements, 0);

  Queue
      .submit([&](sycl::handler &Handler) {
        Handler.set_args(DataPtr, NumOfElements);
        Handler.single_task(UsedKernel);
      })
      .wait();
  int Failed = performResultCheck<NumOfElements>(DataPtr, TestName,
                                                 ExpectedResultValues);
  sycl::free(DataPtr, Queue);
  return Failed;
}

int main() {
  int Failed = 0;
  constexpr size_t N = 256;
  {
    constexpr int ExpectedResultValue = 111;
    std::array<int, N> Numbers;
    std::fill(Numbers.begin(), Numbers.end(), 0);
    setValues<int, 1>(Numbers.data(), Numbers.size(), ExpectedResultValue);
    Failed += performResultCheck(
        N, Numbers.data(),
        "setValues() free function kernel used as normal host function",
        ExpectedResultValue);
  }

  {
    std::array<int, N> Numbers;
    std::iota(Numbers.begin(), Numbers.end(), 0);
    std::array<int, N> ExpectedResultValues;
    std::iota(ExpectedResultValues.begin(), ExpectedResultValues.end(), 0);
    std::reverse(ExpectedResultValues.begin(), ExpectedResultValues.end());
    performReverse(Numbers.data(), Numbers.size());
    Failed += performResultCheck<N>(
        Numbers.data(),
        "performReverse() free function kernel used as normal host function",
        ExpectedResultValues);
  }
  sycl::queue Queue;
  sycl::context Context = Queue.get_context();

  {
    Failed += runSingleTaskKernel<ns::singleTaskKernel<int>, int, N>(
        Queue, Context,
        "performReverse() free function kernel used as device function within "
        "another kernel");
  }

  {
    constexpr int ExpectedResultValue = 222;
    Failed += runNdRangeKernel<ndRangekernel<int, 3>, 3>(
        Queue, Context,
        sycl::nd_range{sycl::range{16, 4, 4}, sycl::range{2, 2, 2}},
        ExpectedResultValue,
        "setValues() free function kernel used as device function within "
        "another kernel");
  }
  return Failed;
}
