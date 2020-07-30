// XFAIL: cuda || level0
// piextUSM*Alloc functions for CUDA are not behaving as described in
// https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/USM/USM.adoc
// https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/USM/cl_intel_unified_shared_memory.asciidoc
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t1.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t1.out
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out

//==---- allocator_vector.cpp - Allocator Container test -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

#include <vector>

using namespace cl::sycl;

const int N = 8;

class foo;
int main() {
  queue q;
  auto dev = q.get_device();
  auto ctxt = q.get_context();

  if (dev.get_info<info::device::usm_host_allocations>()) {
    usm_allocator<int, usm::alloc::host> alloc(ctxt, dev);

    std::vector<int, decltype(alloc)> vec(alloc);
    vec.resize(N);

    for (int i = 0; i < N; i++) {
      vec[i] = i;
    }

    int *res = &vec[0];
    int *vals = &vec[0];

    auto e1 = q.submit([=](handler &h) {
      h.single_task<class foo>([=]() {
        for (int i = 1; i < N; i++) {
          res[0] += vals[i];
        }
      });
    });

    e1.wait();

    int answer = (N * (N - 1)) / 2;

    if (vec[0] != answer)
      return -1;
  }

  if (dev.get_info<info::device::usm_shared_allocations>()) {
    usm_allocator<int, usm::alloc::shared> alloc(ctxt, dev);

    std::vector<int, decltype(alloc)> vec(alloc);
    vec.resize(N);

    for (int i = 0; i < N; i++) {
      vec[i] = i;
    }

    int *res = &vec[0];
    int *vals = &vec[0];

    auto e1 = q.submit([=](handler &h) {
      h.single_task<class bar>([=]() {
        for (int i = 1; i < N; i++) {
          res[0] += vals[i];
        }
      });
    });

    e1.wait();

    int answer = (N * (N - 1)) / 2;

    if (vec[0] != answer)
      return -1;
  }

  if (dev.get_info<info::device::usm_device_allocations>()) {
    usm_allocator<int, usm::alloc::device> alloc(ctxt, dev);

    std::vector<int, decltype(alloc)> vec(alloc);
    vec.resize(N);

    int *res = &vec[0];
    int *vals = &vec[0];

    auto e0 = q.submit([=](handler &h) {
      h.single_task<class baz_init>([=]() {
        res[0] = 0;
        for (int i = 0; i < N; i++) {
          vals[i] = i;
        }
      });
    });

    auto e1 = q.submit([=](handler &h) {
      h.depends_on(e0);
      h.single_task<class baz>([=]() {
        for (int i = 1; i < N; i++) {
          res[0] += vals[i];
        }
      });
    });

    e1.wait();

    int answer = (N * (N - 1)) / 2;
    int result;
    q.memcpy(&result, res, sizeof(int));
    q.wait();

    if (result != answer)
      return -1;
  }

  return 0;
}
