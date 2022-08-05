// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
// Fail is flaky for level_zero, enable when fixed.
// UNSUPPORTED: level_zero

//==--- kernel_info.cpp - SYCL kernel info test ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  queue q;

  buffer<int, 1> buf(range<1>(1));
  auto KernelID = sycl::get_kernel_id<class SingleTask>();
  auto KB =
      get_kernel_bundle<bundle_state::executable>(q.get_context(), {KernelID});
  kernel krn = KB.get_kernel(KernelID);

  q.submit([&](handler &cgh) {
    auto acc = buf.get_access<access::mode::read_write>(cgh);
    cgh.single_task<class SingleTask>(krn, [=]() { acc[0] = acc[0] + 1; });
  });

  const std::string krnName = krn.get_info<info::kernel::function_name>();
  assert(!krnName.empty());
  const cl_uint krnArgCount = krn.get_info<info::kernel::num_args>();
  assert(krnArgCount > 0);
  const context krnCtx = krn.get_info<info::kernel::context>();
  assert(krnCtx == q.get_context());
  const cl_uint krnRefCount = krn.get_info<info::kernel::reference_count>();
  assert(krnRefCount > 0);
  const std::string krnAttr = krn.get_info<info::kernel::attributes>();
  assert(krnAttr.empty());

  device dev = q.get_device();
  const size_t wgSize =
      krn.get_info<info::kernel_device_specific::work_group_size>(dev);
  assert(wgSize > 0);
  const size_t prefWGSizeMult = krn.get_info<
      info::kernel_device_specific::preferred_work_group_size_multiple>(dev);
  assert(prefWGSizeMult > 0);
}
