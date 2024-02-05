// RUN: %clangxx -fsycl -fsycl-device-only -c -o %t.bc %s
// RUN: sycl-post-link %t.bc -spec-const=emulation -S -o %t-split1.txt
// RUN: cat %t-split1_0.ll | FileCheck %s -check-prefixes=CHECK-IR
// RUN: cat %t-split1_0.prop | FileCheck %s -check-prefixes=CHECK-PROP
//
//==----------- SYCL-2020-spec-const-ids-order.cpp -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The test checks that the tool chain correctly identifies all specialization
// constants, emits correct specialization constats map file and can properly
// translate the resulting bitcode to SPIR-V.

#include <sycl/sycl.hpp>

const static sycl::specialization_id<int> SpecConst42{42};
const static sycl::specialization_id<int> SecondValue{42};
const static sycl::specialization_id<int> ConstantId{42};
const static sycl::specialization_id<int> Val23{42};


int main() {
  sycl::queue queue;
  {
    sycl::buffer<int, 1> buf{sycl::range{4}};
    queue.submit([&](sycl::handler &cgh) {
      cgh.set_specialization_constant<SpecConst42>(1);
      cgh.set_specialization_constant<SecondValue>(2);
      cgh.set_specialization_constant<ConstantId>(3);
      cgh.set_specialization_constant<Val23>(4);

      auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
      cgh.single_task<class Kernel3Name>([=](sycl::kernel_handler kh) {
        acc[0] = kh.get_specialization_constant<SpecConst42>();
        acc[1] = kh.get_specialization_constant<SecondValue>();
        acc[2] = kh.get_specialization_constant<ConstantId>();
        acc[3] = kh.get_specialization_constant<Val23>();
      });
    });
  }

  return 0;
}

// CHECK-PROP: [SYCL/specialization constants]
// CHECK-PROP-NEXT: [[UNIQUE_PREFIX:[a-z0-9]+]]____ZL10ConstantId
// CHECK-PROP-NEXT: [[UNIQUE_PREFIX]]____ZL11SecondValue
// CHECK-PROP-NEXT: [[UNIQUE_PREFIX]]____ZL11SpecConst42
// CHECK-PROP-NEXT: [[UNIQUE_PREFIX]]____ZL5Val23

// CHECK-IR: !sycl.specialization-constants = !{![[#MD0:]], ![[#MD1:]], ![[#MD2:]], ![[#MD3:]]}
// CHECK-IR: ![[#MD0]] = !{!"[[UNIQUE_PREFIX:[a-z0-9]+]]____ZL5Val23", i32 [[#ID:]]
// CHECK-IR: ![[#MD1]] = !{!"[[UNIQUE_PREFIX]]____ZL10ConstantId", i32 [[#ID+1]]
// CHECK-IR: ![[#MD2]] = !{!"[[UNIQUE_PREFIX]]____ZL11SecondValue", i32 [[#ID+2]]
// CHECK-IR: ![[#MD3]] = !{!"[[UNIQUE_PREFIX]]____ZL11SpecConst42", i32 [[#ID+3]]
