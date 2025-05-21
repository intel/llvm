// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test verifies whether a vec<T, NumElements> can be passed as a kernel
// parameter to a free function kernel.

#include <sycl/usm.hpp>

#include <sycl/ext/oneapi/free_function_queries.hpp>

#include "helpers.hpp"

static constexpr size_t VEC_SIZE = 3;

static float magnitudeWithoutSqrt(sycl::vec<float, VEC_SIZE> Vec) {
  return Vec.x() * Vec.x() + Vec.y() * Vec.y() + Vec.z() * Vec.z();
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void globalScopeSingleFreeFuncKernel(float *Ptr, size_t NumOfElements,
                                     sycl::vec<float, VEC_SIZE> Vec) {
  for (size_t i = 0; i < NumOfElements; ++i) {
    Ptr[i] = magnitudeWithoutSqrt(Vec);
  }
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void globalScopeNdRangeFreeFuncKernel(float *Ptr,
                                      sycl::vec<float, VEC_SIZE> Vec) {
  size_t Item =
      syclext::this_work_item::get_nd_item<2>().get_global_linear_id();
  Ptr[Item] = magnitudeWithoutSqrt(Vec);
}

namespace ns {
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void nsSingleFreeFuncKernel(float *Ptr, size_t NumOfElements,
                            sycl::vec<float, VEC_SIZE> Vec) {
  for (size_t i = 0; i < NumOfElements; ++i) {
    Ptr[i] = magnitudeWithoutSqrt(Vec);
  }
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<3>))
void nsNdRangeFreeFuncKernel(float *Ptr, sycl::vec<float, VEC_SIZE> Vec) {
  size_t Item =
      syclext::this_work_item::get_nd_item<3>().get_global_linear_id();
  Ptr[Item] = magnitudeWithoutSqrt(Vec);
}
} // namespace ns

// TODO: Need to add checks for a static member functions of a class as free
// function kerenl

int main() {
  int Failed = 0;
  sycl::queue Queue;
  sycl::context Context = Queue.get_context();
  constexpr size_t NumOfElements = 1024;
  float *Data = sycl::malloc_shared<float>(NumOfElements, Queue);

  {
    std::fill(Data, Data + NumOfElements, 0);
    sycl::kernel UsedKernel = getKernel<ns::nsSingleFreeFuncKernel>(Context);

    sycl::vec<float, VEC_SIZE> Vec{1.0, 2.0, 3.0};
    float ExpectedResultValue = magnitudeWithoutSqrt(Vec);
    Queue
        .submit([&](sycl::handler &Handler) {
          Handler.set_args(Data, NumOfElements, Vec);
          Handler.single_task(UsedKernel);
        })
        .wait();

    Failed += performResultCheck(
        NumOfElements, Data, "ns::nsSingleFreeFuncKernel", ExpectedResultValue);
  }

  {
    std::fill(Data, Data + NumOfElements, 0);
    sycl::kernel UsedKernel = getKernel<ns::nsNdRangeFreeFuncKernel>(Context);

    sycl::vec<float, VEC_SIZE> Vec{100.0, 100.0, 100.0};
    float ExpectedResultValue = magnitudeWithoutSqrt(Vec);
    Queue
        .submit([&](sycl::handler &Handler) {
          Handler.set_arg(0, Data);
          Handler.set_arg(1, Vec);
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
    sycl::vec<float, VEC_SIZE> Vec{500.0, 500.0, 500.0};
    float ExpectedResultValue = magnitudeWithoutSqrt(Vec);
    Queue
        .submit([&](sycl::handler &Handler) {
          Handler.set_arg(0, Data);
          Handler.set_arg(1, NumOfElements);
          Handler.set_arg(2, Vec);
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
    sycl::vec<float, VEC_SIZE> Vec{1000.0, 1000.0, 1000.0};
    float ExpectedResultValue = magnitudeWithoutSqrt(Vec);
    Queue
        .submit([&](sycl::handler &Handler) {
          Handler.set_args(Data, Vec);
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
