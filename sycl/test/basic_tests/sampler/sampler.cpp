// REQUIRES: opencl

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out -L %opencl_libs_dir -lOpenCL
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

//==--------------- sampler.cpp - SYCL sampler basic test ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <CL/sycl/context.hpp>
#include <cassert>

namespace sycl {
using namespace cl::sycl;
}

int main() {
  // Check constructor from enums
  sycl::sampler A(sycl::coordinate_normalization_mode::unnormalized,
                  sycl::addressing_mode::clamp, sycl::filtering_mode::nearest);
  assert(A.get_addressing_mode() == sycl::addressing_mode::clamp);
  assert(A.get_coordinate_normalization_mode() ==
         sycl::coordinate_normalization_mode::unnormalized);
  assert(A.get_filtering_mode() == sycl::filtering_mode::nearest);

  sycl::queue Queue;
  sycl::sampler B(A);

  // Check copy constructor
  assert(A.get_addressing_mode() == B.get_addressing_mode() &&
         A.get_coordinate_normalization_mode() ==
             B.get_coordinate_normalization_mode() &&
         A.get_filtering_mode() == B.get_filtering_mode());

  // Check assignment operator
  if (!Queue.is_host()) {
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

    CHECK_OCL_CODE(Err);
    B = sycl::sampler(ClSampler, Queue.get_context());
  } else {
    // Host sampler
    B = sycl::sampler(sycl::coordinate_normalization_mode::normalized,
                      sycl::addressing_mode::repeat,
                      sycl::filtering_mode::linear);
  }
  assert(B.get_addressing_mode() == sycl::addressing_mode::repeat);
  assert(B.get_coordinate_normalization_mode() ==
         sycl::coordinate_normalization_mode::normalized);
  assert(B.get_filtering_mode() == sycl::filtering_mode::linear);

  // Check hasher
  sycl::hash_class<cl::sycl::sampler> Hasher;
  assert(Hasher(A) != Hasher(B));

  // Check move assignment
  sycl::sampler C(B);
  A = std::move(B);
  assert(Hasher(C) == Hasher(A));
  assert(C == A);
  assert(Hasher(C) != Hasher(B));

  // Device sampler.
  {
    sycl::queue Queue;
    Queue.submit([&](sycl::handler &cgh) {
      cgh.single_task<class kernel>([=]() {
        sycl::sampler C = A;
        sycl::sampler D(C);
      });
    });
  }
  return 0;
}
