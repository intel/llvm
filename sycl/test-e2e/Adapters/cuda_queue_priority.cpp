// REQUIRES: gpu, cuda, cuda_dev_kit
// RUN: %{build} %cuda_options -o %t.out
// RUN: %{run} %t.out
//
// Check that queue priority is passed to CUDA runtime
#define SYCL_EXT_ONEAPI_BACKEND_CUDA_EXPERIMENTAL 1
#include <sycl/backend.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/backend/cuda.hpp>
#include <sycl/properties/all_properties.hpp>

#include <cuda.h>

#include <assert.h>

int get_real_priority(sycl::context &C, sycl::device &D,
                      sycl::property_list Props) {
  sycl::queue Q(C, D, Props);
  CUstream QNative = sycl::get_native<sycl::backend::ext_oneapi_cuda>(Q);
  int P;
  CUresult Result = cuStreamGetPriority(QNative, &P);
  assert(Result == CUDA_SUCCESS && "cuStreamGetPriority call failed");
  return P;
}

int main(int Argc, const char *Argv[]) {
  sycl::device D;
  sycl::context C{D};

  int PrioDefault = get_real_priority(C, D, sycl::property_list{});
  int PrioNormal = get_real_priority(
      C, D, {sycl::ext::oneapi::property::queue::priority_normal{}});
  int PrioHigh = get_real_priority(
      C, D, {sycl::ext::oneapi::property::queue::priority_high{}});
  int PrioLow = get_real_priority(
      C, D, {sycl::ext::oneapi::property::queue::priority_low{}});
  // Lower value means higher priority
  assert(PrioDefault == PrioNormal &&
         "priority_normal is not the same as default");
  assert(PrioHigh <= PrioNormal &&
         "priority_high is lower than priority_normal");
  assert(PrioLow >= PrioNormal &&
         "priority_low is higher than priority_normal");
  assert(PrioLow > PrioHigh && "priority_low is the same as priority_high");

  std::cout << "The test passed." << std::endl;
  return 0;
}
