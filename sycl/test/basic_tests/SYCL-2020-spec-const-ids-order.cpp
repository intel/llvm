// RUN: %clangxx -fsycl -fsycl-device-only -c -o %t.bc %s
// RUN: sycl-post-link %t.bc -spec-const=default -S -o %t-split1.txt
// RUN: cat %t-split1_0.ll | FileCheck %s -check-prefixes=CHECK,CHECK-IR
// RUN: cat %t-split1_0.prop | FileCheck %s -check-prefixes=CHECK,CHECK-PROP
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

#include <CL/sycl.hpp>

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
// CHECK: _ZTSN2cl4sycl6detail32specialization_id_name_generatorIL_ZL5Val23EEE
// CHECK-IR-SAME: i32 [[#ID:]]
// CHECK-NEXT: _ZTSN2cl4sycl6detail32specialization_id_name_generatorIL_ZL10ConstantIdEEE
// CHECK-IR-SAME: i32 [[#ID+1]]
// CHECK-NEXT: _ZTSN2cl4sycl6detail32specialization_id_name_generatorIL_ZL11SecondValueEEE
// CHECK-IR-SAME: i32 [[#ID+2]]
// CHECK-NEXT: _ZTSN2cl4sycl6detail32specialization_id_name_generatorIL_ZL11SpecConst42EEE
// CHECK-IR-SAME: i32 [[#ID+3]]
