// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out

//==--- kernel_info_host.cpp - SYCL kernel host info test ------------------==//
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
  assert(krnArgCount == 0);
  const context krnCtx = krn.get_info<info::kernel::context>();
  assert(krnCtx == q.get_context());
  const program krnPrg = krn.get_info<info::kernel::program>();
  assert(krnPrg == prg);

  try {
    krn.get_info<info::kernel::reference_count>();
    assert(false);
  } catch (const sycl::invalid_object_error &) {
  }

  const string_class krnAttr = krn.get_info<info::kernel::attributes>();
  assert(krnAttr.empty());
}
