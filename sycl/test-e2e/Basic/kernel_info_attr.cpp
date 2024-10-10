// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// Fail is flaky for level_zero, enable when fixed.
// UNSUPPORTED: level_zero
//
// Consistently fails with opencl gpu, enable when fixed.
// XFAIL: opencl && gpu
// XFAIL-TRACKER: GSD-8971

//==--- kernel_info_attr.cpp - SYCL info::kernel::attributes test ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------===//

#include <cassert>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/get_kernel_info.hpp>

using namespace sycl;
namespace syclex = sycl::ext::oneapi;

int main() {
  queue q;
  auto ctx = q.get_context();
  buffer<int, 1> buf(range<1>(1));
  auto KernelID = sycl::get_kernel_id<class SingleTask>();
  auto KB = get_kernel_bundle<bundle_state::executable>(ctx, {KernelID});
  kernel krn = KB.get_kernel(KernelID);

  q.submit([&](handler &cgh) {
    auto acc = buf.get_access<access::mode::read_write>(cgh);
    cgh.single_task<class SingleTask>(krn, [=]() { acc[0] = acc[0] + 1; });
  });

  const std::string krnAttr = krn.get_info<info::kernel::attributes>();
  assert(krnAttr.empty());
  const std::string krnAttrExt =
      syclex::get_kernel_info<SingleTask, info::kernel::attributes>(ctx);
  assert(krnAttr == krnAttrExt);
  return 0;
}
