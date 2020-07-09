// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out -L %opencl_libs_dir -lOpenCL
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// XFAIL: level0
// "die: piextKernelSetArgSampler: not implemented"

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

struct SamplerWrapper {
  SamplerWrapper(sycl::coordinate_normalization_mode Norm,
                 sycl::addressing_mode Addr, sycl::filtering_mode Filter)
      : Smpl(Norm, Addr, Filter), A(0) {}

  sycl::sampler Smpl;
  int A;
};

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
  B = sycl::sampler(sycl::coordinate_normalization_mode::normalized,
                    sycl::addressing_mode::repeat,
                    sycl::filtering_mode::linear);

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

  SamplerWrapper WrappedSmplr(
      sycl::coordinate_normalization_mode::normalized,
      sycl::addressing_mode::repeat, sycl::filtering_mode::linear);

  // Device sampler.
  {
    sycl::queue Queue;
    Queue.submit([&](sycl::handler &cgh) {
      cgh.single_task<class kernel>([=]() {
        sycl::sampler C = A;
        sycl::sampler D(C);
        sycl::sampler E(WrappedSmplr.Smpl);
      });
    });
  }
  return 0;
}
