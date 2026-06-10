// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
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

#include <algorithm>
#include <cassert>
#include <cctype>
#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>

#include <sycl/ext/oneapi/get_kernel_info.hpp>

using namespace sycl;
namespace syclex = sycl::ext::oneapi;

class SingleTask;

int main() {
  queue q;
  auto ctx = q.get_context();
  buffer<int, 1> buf(range<1>(1));
  auto KernelID = sycl::get_kernel_id<SingleTask>();
  auto KB = get_kernel_bundle<bundle_state::executable>(ctx, {KernelID});
  kernel krn = KB.get_kernel(KernelID);

  q.submit([&](handler &cgh) {
    auto acc = buf.get_access<access::mode::read_write>(cgh);
    cgh.single_task<SingleTask>([=]() { acc[0] = acc[0] + 1; });
  });

  auto removeWhitespace = [](std::string Attr) {
    Attr.erase(std::remove_if(Attr.begin(), Attr.end(),
                              [](unsigned char C) { return std::isspace(C); }),
               Attr.end());
    return Attr;
  };

  const std::string krnAttr = krn.get_info<info::kernel::attributes>();
  const std::string normalizedKrnAttr = removeWhitespace(krnAttr);
  assert(normalizedKrnAttr.empty() ||
         normalizedKrnAttr.find("intel_reqd_workgroup_walk_order(") !=
             std::string::npos);

  const std::string krnAttrExt =
      syclex::get_kernel_info<SingleTask, info::kernel::attributes>(ctx);
  assert(normalizedKrnAttr == removeWhitespace(krnAttrExt));
  return 0;
}
