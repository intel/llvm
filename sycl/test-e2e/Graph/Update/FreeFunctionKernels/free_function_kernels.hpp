#include "../../graph_common.hpp"
#include "sycl/ext/oneapi/kernel_properties/properties.hpp"
#include "sycl/kernel_bundle.hpp"
#include <sycl/ext/oneapi/free_function_queries.hpp>

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((exp_ext::single_task_kernel))
void ff_0(int *ptr) {
  for (size_t i{0}; i < Size; ++i) {
    ptr[i] = i;
  }
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((exp_ext::single_task_kernel))
void ff_1(int *ptr) {
  for (size_t i{0}; i < Size; ++i) {
    ptr[i] += i;
  }
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((exp_ext::single_task_kernel))
void ff_2(int *ptr, size_t size, size_t numKernelLoops) {
  for (size_t j = 0; j < numKernelLoops; j++) {
    for (size_t i = 0; i < size; i++) {
      ptr[i] += i;
    }
  }
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((exp_ext::nd_range_kernel<3>))
void ff_3(int *ptr) {
  size_t GlobalID =
      ext::oneapi::this_work_item::get_nd_item<3>().get_global_linear_id();
  ptr[GlobalID] = GlobalID;
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((exp_ext::nd_range_kernel<3>))
void ff_4(int *ptr) {
  size_t GlobalID =
      ext::oneapi::this_work_item::get_nd_item<3>().get_global_linear_id();
  ptr[GlobalID] *= 2;
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((exp_ext::nd_range_kernel<1>))
void ff_5(int *ptrA, int *ptrB, int *ptrC) {
  size_t id = ext::oneapi::this_work_item::get_nd_item<1>().get_global_id();
  ptrC[id] += ptrA[id] * ptrB[id];
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((exp_ext::single_task_kernel))
void ff_6(int *ptr, int scalarValue) {
  for (size_t i{0}; i < Size; ++i) {
    ptr[i] = scalarValue;
  }
}
