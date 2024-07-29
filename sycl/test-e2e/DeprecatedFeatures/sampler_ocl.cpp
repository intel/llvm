// REQUIRES: opencl, opencl_icd

// RUN: %{build} -D__SYCL_INTERNAL_API -o %t.out %opencl_lib
// RUN: %{run} %t.out

// TODO: Reenable, see https://github.com/intel/llvm/issues/14679
// UNSUPPORTED: windows, linux

//==--------------- sampler.cpp - SYCL sampler basic test ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <sycl/context.hpp>
#include <sycl/detail/core.hpp>

namespace sycl {
using namespace sycl;
}

int main() {
  sycl::queue Queue;
  sycl::sampler B(sycl::coordinate_normalization_mode::unnormalized,
                  sycl::addressing_mode::clamp, sycl::filtering_mode::nearest);

  // OpenCL sampler
  cl_int Err = CL_SUCCESS;
#ifdef CL_VERSION_2_0
  const cl_sampler_properties sprops[] = {
      CL_SAMPLER_NORMALIZED_COORDS,
      static_cast<cl_sampler_properties>(true),
      CL_SAMPLER_ADDRESSING_MODE,
      static_cast<cl_sampler_properties>(CL_ADDRESS_REPEAT),
      CL_SAMPLER_FILTER_MODE,
      static_cast<cl_sampler_properties>(CL_FILTER_LINEAR),
      0};
  cl_sampler ClSampler =
      clCreateSamplerWithProperties(Queue.get_context().get(), sprops, &Err);
#else
  cl_sampler ClSampler =
      clCreateSampler(Queue.get_context().get(), true, CL_ADDRESS_REPEAT,
                      CL_FILTER_LINEAR, &Err);
#endif
  // If device doesn't support sampler - skip it
  if (Err == CL_INVALID_OPERATION)
    return 0;

  assert(Err == CL_SUCCESS);
  B = sycl::sampler(ClSampler, Queue.get_context());

  assert(B.get_addressing_mode() == sycl::addressing_mode::repeat);
  assert(B.get_coordinate_normalization_mode() ==
         sycl::coordinate_normalization_mode::normalized);
  assert(B.get_filtering_mode() == sycl::filtering_mode::linear);
}
