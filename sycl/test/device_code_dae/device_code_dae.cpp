// NOTE A temporary test before this compilation flow is enabled by default in
// driver
// UNSUPPORTED: cuda
// CUDA does not support SPIR-V.
// RUN: %clangxx -fsycl-device-only -Xclang -fenable-sycl-dae -Xclang -fsycl-int-header=int_header.h %s -c -o device_code.bc -I %sycl_include -Wno-sycl-strict
// RUN: %clangxx -include int_header.h -g -c %s -o host_code.o -I %sycl_include -Wno-sycl-strict
// RUN: llvm-link -o=linked_device_code.bc device_code.bc
// RUN: sycl-post-link -emit-param-info linked_device_code.bc
// RUN: llvm-spirv -o linked_device_code.spv linked_device_code.bc
// RUN: echo -e -n "[Code|Properties]\nlinked_device_code.spv|linked_device_code_0.prop" > table.txt
// RUN: clang-offload-wrapper -o wrapper.bc -host=x86_64 -kind=sycl -target=spir64 -batch table.txt
// RUN: %clangxx -c wrapper.bc -o wrapper.o
// RUN: %clangxx wrapper.o host_code.o -o app.exe -lsycl
// RUN: env SYCL_BE=%sycl_be ./app.exe

//==---------device_code_dae.cpp - dead argument elimination test ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

#include <cassert>

class KernelNameA;
class KernelNameB;
class KernelNameC;
using namespace cl::sycl;

void verifyAndReset(buffer<int, 1> buf, int expected) {
  auto acc = buf.get_access<access::mode::read_write>();
  assert(acc[0] == expected);
  acc[0] = 0;
}

int main() {
  buffer<int, 1> buf{range<1>(1)};
  int gold = 42;
  queue q;

  q.submit([&](handler &cgh) {
    auto acc = buf.get_access<access::mode::read_write>(cgh);
    cgh.single_task<KernelNameA>([=]() { acc[0] = gold; });
  });

  verifyAndReset(buf, gold);

  // Check usage of program class
  program prgB{q.get_context()};
  prgB.build_with_kernel_type<KernelNameB>();
  kernel krnB = prgB.get_kernel<KernelNameB>();
  q.submit([&](handler &cgh) {
    auto acc = buf.get_access<access::mode::read_write>(cgh);
    cgh.single_task<KernelNameB>(krnB, [=]() { acc[0] = gold; });
  });

  verifyAndReset(buf, gold);

  // Check the non-cacheable case
  program prgC{q.get_context()};
  prgC.compile_with_kernel_type<KernelNameC>();
  prgC.link();
  kernel krnC = prgC.get_kernel<KernelNameC>();
  q.submit([&](handler &cgh) {
    auto acc = buf.get_access<access::mode::read_write>(cgh);
    cgh.single_task<KernelNameC>(krnC, [=]() { acc[0] = gold; });
  });

  verifyAndReset(buf, gold);
}
