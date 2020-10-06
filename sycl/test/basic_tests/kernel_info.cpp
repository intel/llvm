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

#include <CL/sycl.hpp>
#include <cassert>

using namespace cl::sycl;

int main() {
  queue q;

  buffer<int, 1> buf(range<1>(1));
  program prg(q.get_context());

  prg.build_with_kernel_type<class SingleTask>();
  assert(prg.has_kernel<class SingleTask>());
  kernel krn = prg.get_kernel<class SingleTask>();

  q.submit([&](handler &cgh) {
    auto acc = buf.get_access<access::mode::read_write>(cgh);
    cgh.single_task<class SingleTask>(krn, [=]() { acc[0] = acc[0] + 1; });
  });

  const string_class krnName = krn.get_info<info::kernel::function_name>();
  assert(!krnName.empty());
  const cl_uint krnArgCount = krn.get_info<info::kernel::num_args>();
  assert(krnArgCount > 0);
  const context krnCtx = krn.get_info<info::kernel::context>();
  assert(krnCtx == q.get_context());
  const program krnPrg = krn.get_info<info::kernel::program>();
  assert(krnPrg == prg);
  const cl_uint krnRefCount = krn.get_info<info::kernel::reference_count>();
  assert(krnRefCount > 0);
  const string_class krnAttr = krn.get_info<info::kernel::attributes>();
  assert(krnAttr.empty());

  device dev = q.get_device();
  const size_t wgSize =
      krn.get_work_group_info<info::kernel_work_group::work_group_size>(dev);
  assert(wgSize > 0);
  const size_t wgSizeNew =
      krn.get_info<info::kernel_device_specific::work_group_size>(dev);
  assert(wgSizeNew > 0);
  assert(wgSize == wgSizeNew);
  const size_t prefWGSizeMult = krn.get_work_group_info<
      info::kernel_work_group::preferred_work_group_size_multiple>(dev);
  assert(prefWGSizeMult > 0);
  const size_t prefWGSizeMultNew = krn.get_info<
      info::kernel_device_specific::preferred_work_group_size_multiple>(dev);
  assert(prefWGSizeMultNew > 0);
  assert(prefWGSizeMult == prefWGSizeMultNew);
  const cl_ulong prvMemSize =
      krn.get_work_group_info<info::kernel_work_group::private_mem_size>(dev);
  assert(prvMemSize == 0);
  const cl_ulong prvMemSizeNew =
      krn.get_info<info::kernel_device_specific::private_mem_size>(dev);
  assert(prvMemSizeNew == 0);
}
