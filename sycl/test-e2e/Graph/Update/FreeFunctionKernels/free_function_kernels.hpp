#pragma once

#include "../../graph_common.hpp"
#include "sycl/ext/oneapi/kernel_properties/properties.hpp"
#include "sycl/kernel_bundle.hpp"
#include <sycl/ext/oneapi/free_function_queries.hpp>

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((exp_ext::single_task_kernel))
void ff_0(int *Ptr) {
  for (size_t i{0}; i < Size; ++i) {
    Ptr[i] = i;
  }
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((exp_ext::single_task_kernel))
void ff_1(int *Ptr) {
  for (size_t i{0}; i < Size; ++i) {
    Ptr[i] += i;
  }
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((exp_ext::single_task_kernel))
void ff_2(int *Ptr, size_t Size, size_t NumKernelLoops) {
  for (size_t j{0}; j < NumKernelLoops; j++) {
    for (size_t i{0}; i < Size; i++) {
      Ptr[i] += i;
    }
  }
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((exp_ext::nd_range_kernel<3>))
void ff_3(int *Ptr) {
  size_t GlobalID =
      ext::oneapi::this_work_item::get_nd_item<3>().get_global_linear_id();
  Ptr[GlobalID] = GlobalID;
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((exp_ext::nd_range_kernel<3>))
void ff_4(int *Ptr) {
  size_t GlobalID =
      ext::oneapi::this_work_item::get_nd_item<3>().get_global_linear_id();
  Ptr[GlobalID] *= 2;
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((exp_ext::nd_range_kernel<1>))
void ff_5(int *PtrA, int *PtrB, int *PtrC) {
  size_t GlobalID =
      ext::oneapi::this_work_item::get_nd_item<1>().get_global_id();
  PtrC[GlobalID] += PtrA[GlobalID] * PtrB[GlobalID];
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((exp_ext::single_task_kernel))
void ff_6(int *Ptr, int ScalarValue) {
  for (size_t i{0}; i < Size; ++i) {
    Ptr[i] = ScalarValue;
  }
}
