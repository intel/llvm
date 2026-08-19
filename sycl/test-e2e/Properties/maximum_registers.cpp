//==----------- maximum_registers.cpp  - DPC++ SYCL on-device test --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This test verifies the effect of the "maximum_registers<num>" and
// "maximum_registers_automatic" kernel properties from
// sycl_ext_intel_maximum_registers in device code and checks the computed
// results on the host. It exercises the property across these dimensions:
//   * SYCL and ESIMD kernels.
//   * Free function kernels and lambda kernels.
//   * maximum_registers<num> and maximum_registers_automatic.

// REQUIRES: arch-intel_gpu_bmg_g21

// XFAIL: run-mode
// XFAIL-TRACKER: GSD-4149

// UNSUPPORTED: spirv-backend
// UNSUPPORTED-INTENDED: The required SPIR-V extensions are not supported.

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// TODO: Add AOT test when removing XFAIL

#include "../helpers.hpp"
#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/maximum_registers_properties.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/usm.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental;
namespace syclexp = sycl::ext::oneapi::experimental;
namespace intelexp = sycl::ext::intel::experimental;

constexpr unsigned Size = 32;
constexpr unsigned VL = 16;

bool checkResult(const std::vector<float> &A, int Inc, const char *Msg) {
  int err_cnt = 0;
  unsigned Sz = A.size();

  for (unsigned i = 0; i < Sz; ++i) {
    if (A[i] != i + Inc)
      if (++err_cnt < 10)
        std::cerr << "failed at A[" << i << "]: " << A[i] << " != " << i + Inc
                  << "\n";
  }

  if (err_cnt > 0) {
    std::cout << Msg << " failed. pass rate: "
              << ((float)(Sz - err_cnt) / (float)Sz) * 100.0f << "% ("
              << (Sz - err_cnt) << "/" << Sz << ")\n";
    return false;
  }
  std::cout << Msg << " passed\n";
  return true;
}

// Free function kernels. The property is attached at the definition via
// SYCL_EXT_ONEAPI_FUNCTION_PROPERTY.
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((intelexp::maximum_registers<256>))
void free_function_kernel_specified(float *Ptr) {
  size_t i = ext::oneapi::this_work_item::get_nd_item<1>().get_global_id(0);
  Ptr[i] += 1;
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((intelexp::maximum_registers_automatic))
void free_function_kernel_automatic(float *Ptr) {
  size_t i = ext::oneapi::this_work_item::get_nd_item<1>().get_global_id(0);
  Ptr[i] += 1;
}

// ESIMD free function kernels. The property is attached at the definition and
// the function is marked as an ESIMD kernel.
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((intelexp::maximum_registers<256>))
void esimd_free_function_kernel_specified(float *Ptr) SYCL_ESIMD_KERNEL {
  size_t i = ext::oneapi::this_work_item::get_nd_item<1>().get_global_id(0);
  float *Base = Ptr + i * VL;
  simd<float, VL> va;
  va.copy_from(Base);
  simd<float, VL> vc = va + 1;
  vc.copy_to(Base);
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((intelexp::maximum_registers_automatic))
void esimd_free_function_kernel_automatic(float *Ptr) SYCL_ESIMD_KERNEL {
  size_t i = ext::oneapi::this_work_item::get_nd_item<1>().get_global_id(0);
  float *Base = Ptr + i * VL;
  simd<float, VL> va;
  va.copy_from(Base);
  simd<float, VL> vc = va + 1;
  vc.copy_to(Base);
}

// SYCL lambda kernel. The property is attached via a launch_config.
template <typename PropsT>
bool runLambdaSYCL(queue &q, PropsT Props, const char *Msg) {
  std::vector<float> A(Size);
  for (unsigned i = 0; i < Size; ++i)
    A[i] = i;
  float *Ptr = malloc_shared<float>(Size, q);
  try {
    for (unsigned i = 0; i < Size; ++i)
      Ptr[i] = A[i];

    syclexp::parallel_for(q, syclexp::launch_config{range<1>{Size}, Props},
                          [=](id<1> i) { Ptr[i] += 1; });
    q.wait();

    for (unsigned i = 0; i < Size; ++i)
      A[i] = Ptr[i];
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    free(Ptr, q);
    return false;
  }
  free(Ptr, q);
  return checkResult(A, 1, Msg);
}

// ESIMD lambda kernel. The property is attached via a launch_config.
template <typename PropsT>
bool runLambdaESIMD(queue &q, PropsT Props, const char *Msg) {
  std::vector<float> A(Size);
  for (unsigned i = 0; i < Size; ++i)
    A[i] = i;
  float *Ptr = malloc_shared<float>(Size, q);
  try {
    for (unsigned i = 0; i < Size; ++i)
      Ptr[i] = A[i];

    syclexp::parallel_for(q, syclexp::launch_config{range<1>{Size / VL}, Props},
                          [=](id<1> i) SYCL_ESIMD_KERNEL {
                            float *Base = Ptr + i * VL;
                            simd<float, VL> va;
                            va.copy_from(Base);
                            simd<float, VL> vc = va + 1;
                            vc.copy_to(Base);
                          });
    q.wait();

    for (unsigned i = 0; i < Size; ++i)
      A[i] = Ptr[i];
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    free(Ptr, q);
    return false;
  }
  free(Ptr, q);
  return checkResult(A, 1, Msg);
}

// Free function kernel. The property is attached at the definition. GlobalSize
// is the number of work-items to launch (Size for SYCL, Size / VL for ESIMD,
// where each work-item processes a VL-wide block).
template <auto *KernelFn>
bool runFreeFunction(queue &q, unsigned GlobalSize, const char *Msg) {
  std::vector<float> A(Size);
  for (unsigned i = 0; i < Size; ++i)
    A[i] = i;
  float *Ptr = malloc_shared<float>(Size, q);
  try {
    for (unsigned i = 0; i < Size; ++i)
      Ptr[i] = A[i];

    syclexp::nd_launch(q,
                       nd_range<1>{range<1>{GlobalSize}, range<1>{GlobalSize}},
                       syclexp::kernel_function<KernelFn>, Ptr);
    q.wait();

    for (unsigned i = 0; i < Size; ++i)
      A[i] = Ptr[i];
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    free(Ptr, q);
    return false;
  }
  free(Ptr, q);
  return checkResult(A, 1, Msg);
}

int main(void) {
  queue q(sycl::gpu_selector_v, exceptionHandlerHelper);

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";

  syclexp::properties specified_props{maximum_registers<256>};
  syclexp::properties automatic_props{maximum_registers_automatic};

  bool Pass = true;

  // SYCL lambda kernels.
  Pass &=
      runLambdaSYCL(q, specified_props, "SYCL lambda maximum_registers<256>");
  Pass &= runLambdaSYCL(q, automatic_props,
                        "SYCL lambda maximum_registers_automatic");

  // ESIMD lambda kernels.
  Pass &=
      runLambdaESIMD(q, specified_props, "ESIMD lambda maximum_registers<256>");
  Pass &= runLambdaESIMD(q, automatic_props,
                         "ESIMD lambda maximum_registers_automatic");

  // SYCL free function kernels (property attached at the definition).
  Pass &= runFreeFunction<free_function_kernel_specified>(
      q, Size, "SYCL free function maximum_registers<256>");
  Pass &= runFreeFunction<free_function_kernel_automatic>(
      q, Size, "SYCL free function maximum_registers_automatic");

  // ESIMD free function kernels (property attached at the definition).
  Pass &= runFreeFunction<esimd_free_function_kernel_specified>(
      q, Size / VL, "ESIMD free function maximum_registers<256>");
  Pass &= runFreeFunction<esimd_free_function_kernel_automatic>(
      q, Size / VL, "ESIMD free function maximum_registers_automatic");

  if (!Pass) {
    std::cout << "Test failed\n";
    return 1;
  }

  std::cout << "Test passed\n";
  return 0;
}
