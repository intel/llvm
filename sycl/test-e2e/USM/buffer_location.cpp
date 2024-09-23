// UNSUPPORTED: gpu
// RUN: %{build} -o %t1.out
// RUN: %{run} %t1.out

//==------------------- buffer_location.cpp - USM buffer location ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <memory>

#include <sycl/detail/core.hpp>

#include <sycl/ext/intel/experimental/usm_properties.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

// Pointer wrapper allows custom deleter to clean up resources
struct ptr_wrapper {
  sycl::context m_ctx;
  int *m_ptr;
  ptr_wrapper(sycl::context ctx, int *ptr) {
    m_ctx = ctx;
    m_ptr = ptr;
  }
  ~ptr_wrapper() { sycl::free(m_ptr, m_ctx); }
};

int main() {
  queue q;
  auto dev = q.get_device();
  auto ctxt = q.get_context();

  if (!dev.get_info<info::device::usm_device_allocations>())
    return 0;

  int true_buf_loc = 3;
  auto test_src_ptr_unq = std::unique_ptr<ptr_wrapper>(new ptr_wrapper(
      ctxt, malloc_device<int>(
                1, dev, ctxt,
                property_list{
                    ext::intel::experimental::property::usm::buffer_location(
                        true_buf_loc)})));
  if (test_src_ptr_unq == nullptr) {
    return -1;
  }
  auto test_dst_ptr_unq = std::unique_ptr<ptr_wrapper>(new ptr_wrapper(
      ctxt, malloc_device<int>(
                1, dev, ctxt,
                property_list{
                    ext::intel::experimental::property::usm::buffer_location(
                        true_buf_loc)})));
  if (test_dst_ptr_unq == nullptr) {
    return -1;
  }
  int src_val = 3;
  int dst_val = 4;
  int *test_src_ptr = test_src_ptr_unq.get()->m_ptr;
  int *test_dst_ptr = test_dst_ptr_unq.get()->m_ptr;
  event e0 = q.memcpy(test_src_ptr, &src_val, sizeof(int));
  e0.wait();
  event e1 = q.memcpy(test_dst_ptr, &dst_val, sizeof(int));
  e1.wait();

  auto e2 = q.submit([=](handler &cgh) {
    cgh.single_task<class foo>([=]() { *test_src_ptr = *test_dst_ptr; });
  });
  e2.wait();

  if (*test_src_ptr != dst_val) {
    return -1;
  }

  return 0;
}
