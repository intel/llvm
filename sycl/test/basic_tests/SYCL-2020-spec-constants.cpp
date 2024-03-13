// RUN: %clangxx -fsycl -fsycl-device-only -c -o %t.bc %s
// RUN: %if asserts %{sycl-post-link -debug-only=SpecConst %t.bc -spec-const=native -o %t-split1.txt 2>&1 | FileCheck %s -check-prefixes=CHECK-LOG %} %else %{sycl-post-link %t.bc -spec-const=native -o %t-split1.txt 2>&1 %}
// RUN: cat %t-split1_0.prop | FileCheck %s -check-prefixes=CHECK,CHECK-RT
// RUN: sycl-post-link %t.bc -spec-const=emulation -o %t-split2.txt
// RUN: cat %t-split2_0.prop | FileCheck %s -check-prefixes=CHECK,CHECK-DEF
// RUN: llvm-spirv -o %t-split1_0.spv -spirv-max-version=1.1 -spirv-ext=+all %t-split1_0.bc
// RUN: llvm-spirv -o - --to-text %t-split1_0.spv | FileCheck %s -check-prefixes=CHECK-SPV
//
//==----------- SYCL-2020-spec-constants.cpp -------------------------------==//
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

#include <cstdint>
#include <vector>

// FIXME: there is a bug with generation SPIR-V friendly IR for boolean spec
// constants, which was likely interoduced by https://github.com/intel/llvm/pull/3513
// constexpr sycl::specialization_id<bool> bool_id;
constexpr sycl::specialization_id<int8_t> int8_id(42);
constexpr sycl::specialization_id<uint8_t> uint8_id(26);
constexpr sycl::specialization_id<int16_t> int16_id(34);
constexpr sycl::specialization_id<uint16_t> uint16_id(14);
constexpr sycl::specialization_id<int32_t> int32_id(52);
constexpr sycl::specialization_id<uint32_t> uint32_id(46);
constexpr sycl::specialization_id<int64_t> int64_id(27);
constexpr sycl::specialization_id<uint64_t> uint64_id(81);
// FIXME: enable once we support constexpr constructor for half
// constexpr sycl::specialization_id<half> half_id(9.0);
constexpr sycl::specialization_id<float> float_id(710.0);
constexpr sycl::specialization_id<double> double_id(11.0);
constexpr sycl::marray<double, 5> ma;
constexpr sycl::specialization_id<sycl::marray<double, 5>> marray_id5(151.0);
constexpr sycl::specialization_id<sycl::marray<double, 1>> marray_id1(116.0);
constexpr sycl::specialization_id<sycl::marray<double, 5>> marray_id_def(ma);
constexpr sycl::vec<double, 4> v{};
constexpr sycl::specialization_id<sycl::vec<double, 4>> vec_id_def(v);
constexpr sycl::specialization_id<sycl::vec<double, 1>> vec_id1(211.0);
constexpr sycl::specialization_id<sycl::vec<double, 4>> vec_id4(131.0);

constexpr sycl::vec<long long, 1> vv(1);

template <typename T> inline constexpr auto helper(int x) { return T{x}; }

constexpr sycl::specialization_id<sycl::vec<long long, 1>>
    vec_helper1(helper<sycl::vec<long long, 1>>(1));

struct composite {
  int a;
  int b;
  constexpr composite(int a, int b): a(a), b(b) {}
};

constexpr sycl::specialization_id<composite> composite_id(12, 13);

enum class enumeration { a, b, c };

constexpr sycl::specialization_id<enumeration> enumeration_id(enumeration::c);

class SpecializedKernel;

int main() {
  sycl::queue queue;

  std::vector<float> vec(1);
  {
    sycl::buffer<float, 1> buf(vec.data(), vec.size());
    queue.submit([&](sycl::handler &h) {
      auto acc = buf.get_access<sycl::access::mode::write>(h);

      h.single_task<SpecializedKernel>([=](sycl::kernel_handler kh) {
        // see FIXME above about bool type support
        // auto i1 = kh.get_specialization_constant<bool_id>();
        auto i8 = kh.get_specialization_constant<int8_id>();
        auto u8 = kh.get_specialization_constant<uint8_id>();
        auto i16 = kh.get_specialization_constant<int16_id>();
        auto u16 = kh.get_specialization_constant<uint16_id>();
        auto i32 = kh.get_specialization_constant<int32_id>();
        auto u32 = kh.get_specialization_constant<uint32_id>();
        auto i64 = kh.get_specialization_constant<int64_id>();
        auto u64 = kh.get_specialization_constant<uint64_id>();
        // see FIXME above about half type support
        // auto f16 = kh.get_specialization_constant<half_id>();
        auto f32 = kh.get_specialization_constant<float_id>();
        auto f64 = kh.get_specialization_constant<double_id>();
        auto c = kh.get_specialization_constant<composite_id>();
        auto e = kh.get_specialization_constant<enumeration_id>();

        // see FIXMEs above about bool and half types support
        acc[0] = /*i1 +*/ i8 + u8 + i16 + u16 + i32 + u32 + i64 + u64 +
                 // f16.get() +
                 f32 + f64 + c.a + c.b + static_cast<float>(e);
      });
    });
  }

  return 0;
}

// CHECK: [SYCL/specialization constants]
// CHECK-DAG: [[UNIQUE_PREFIX:[a-z0-9]+]]____ZL12composite_id=2|
// See FIXME above about bool type support
// CHECK-disabled: [[UNIQUE_PREFIX]]____IL_ZL7bool_id=2|
// CHECK-DAG: [[UNIQUE_PREFIX]]____ZL7int8_id=2|
// CHECK-DAG: [[UNIQUE_PREFIX]]____ZL8float_id=2|
// CHECK-DAG: [[UNIQUE_PREFIX]]____ZL8int16_id=2|
// CHECK-DAG: [[UNIQUE_PREFIX]]____ZL8int32_id=2|
// CHECK-DAG: [[UNIQUE_PREFIX]]____ZL8int64_id=2|
// CHECK-DAG: [[UNIQUE_PREFIX]]____ZL8uint8_id=2|
// CHECK-DAG: [[UNIQUE_PREFIX]]____ZL9double_id=2|
// CHECK-DAG: [[UNIQUE_PREFIX]]____ZL9uint16_id=2|
// CHECK-DAG: [[UNIQUE_PREFIX]]____ZL9uint32_id=2|
// CHECK-DAG: [[UNIQUE_PREFIX]]____ZL9uint64_id=2|
// CHECK-DAG: [[UNIQUE_PREFIX]]____ZL14enumeration_id=2|
// FIXME: check line for half constant

// CHECK-RT: [SYCL/specialization constants default values]
// CHECK-DEF: [SYCL/specialization constants default values]
// CHECK-DEF: all=2|

// CHECK-LOG: sycl.specialization-constants
// CHECK-LOG:[[UNIQUE_PREFIX:[a-z0-9]+]]____ZL7int8_id={0, 0, 1}
// CHECK-NEXT-LOG:[[UNIQUE_PREFIX]]____ZL8uint8_id={1, 0, 1}
// CHECK-NEXT-LOG:[[UNIQUE_PREFIX]]____ZL8int16_id={2, 0, 2}
// CHECK-NEXT-LOG:[[UNIQUE_PREFIX]]____ZL9uint16_id={3, 0, 2}
// CHECK-NEXT-LOG:[[UNIQUE_PREFIX]]____ZL8int32_id={4, 0, 4}
// CHECK-NEXT-LOG:[[UNIQUE_PREFIX]]____ZL9uint32_id={5, 0, 4}
// CHECK-NEXT-LOG:[[UNIQUE_PREFIX]]____ZL8int64_id={6, 0, 8}
// CHECK-NEXT-LOG:[[UNIQUE_PREFIX]]____ZL9uint64_id={7, 0, 8}
// CHECK-NEXT-LOG:[[UNIQUE_PREFIX]]____ZL8float_id={8, 0, 4}
// CHECK-NEXT-LOG:[[UNIQUE_PREFIX]]____ZL9double_id={9, 0, 8}
// CHECK-NEXT-LOG:[[UNIQUE_PREFIX]]____ZL12composite_id={10, 0, 4}
// CHECK-NEXT-LOG:[[UNIQUE_PREFIX]]____ZL12composite_id={11, 4, 4}
// CHECK-NEXT-LOG:[[UNIQUE_PREFIX]]____ZL14enumeration_id={12, 0, 4}
// CHECK-NEXT-LOG:{0, 1, 42}
// CHECK-NEXT-LOG:{1, 1, 26}
// CHECK-NEXT-LOG:{2, 2, 34}
// CHECK-NEXT-LOG:{4, 2, 14}
// CHECK-NEXT-LOG:{6, 4, 52}
// CHECK-NEXT-LOG:{10, 4, 46}
// CHECK-NEXT-LOG:{14, 8, 27}
// CHECK-NEXT-LOG:{22, 8, 81}
// CHECK-NEXT-LOG:{30, 4, 7.100000e+02}
// CHECK-NEXT-LOG:{34, 8, 1.100000e+01}
// CHECK-NEXT-LOG:{42, 4, 12}
// CHECK-NEXT-LOG:{46, 4, 13}
// CHECK-NEXT-LOG:{50, 4, 2}

// CHECK-SPV-DAG: Decorate [[#SPEC0:]] SpecId 0
// CHECK-SPV-DAG: Decorate [[#SPEC1:]] SpecId 1
// CHECK-SPV-DAG: Decorate [[#SPEC2:]] SpecId 2
// CHECK-SPV-DAG: Decorate [[#SPEC3:]] SpecId 3
// CHECK-SPV-DAG: Decorate [[#SPEC4:]] SpecId 4
// CHECK-SPV-DAG: Decorate [[#SPEC5:]] SpecId 5
// CHECK-SPV-DAG: Decorate [[#SPEC6:]] SpecId 6
// CHECK-SPV-DAG: Decorate [[#SPEC7:]] SpecId 7
// CHECK-SPV-DAG: Decorate [[#SPEC8:]] SpecId 8
// CHECK-SPV-DAG: Decorate [[#SPEC9:]] SpecId 9
// CHECK-SPV-DAG: Decorate [[#SPEC10:]] SpecId 10
// CHECK-SPV-DAG: Decorate [[#SPEC11:]] SpecId 11
// CHECK-SPV-DAG: Decorate [[#SPEC12:]] SpecId 12

// CHECK-SPV-DAG: TypeInt [[#I64TY:]] 64 0
// CHECK-SPV-DAG: TypeInt [[#I32TY:]] 32 0
// CHECK-SPV-DAG: TypeInt [[#I8TY:]]  8 0
// CHECK-SPV-DAG: TypeInt [[#I16TY:]] 16 0
// CHECK-SPV-DAG: TypeFloat [[#F32TY:]] 32
// CHECK-SPV-DAG: TypeFloat [[#F64TY:]] 64
// CHECK-SPV-DAG: TypeStruct [[#COMPTY:]] [[#I32TY]] [[#I32TY]]
// CHECK-SPV-DAG: SpecConstant [[#I8TY]]  [[#SPEC0]] 42
// CHECK-SPV-DAG: SpecConstant [[#I8TY]]  [[#SPEC1]] 26
// CHECK-SPV-DAG: SpecConstant [[#I16TY]] [[#SPEC2]] 34
// CHECK-SPV-DAG: SpecConstant [[#I16TY]] [[#SPEC3]] 14
// CHECK-SPV-DAG: SpecConstant [[#I32TY]] [[#SPEC4]] 52
// CHECK-SPV-DAG: SpecConstant [[#I32TY]] [[#SPEC5]] 46
// CHECK-SPV-DAG: SpecConstant [[#I64TY]] [[#SPEC6]] 27 0
// CHECK-SPV-DAG: SpecConstant [[#I64TY]] [[#SPEC7]] 81 0
// CHECK-SPV-DAG: SpecConstant [[#F32TY]] [[#SPEC8]] 1144094720
// CHECK-SPV-DAG: SpecConstant [[#F64TY]] [[#SPEC9]] 0 1076232192
// CHECK-SPV-DAG: SpecConstant [[#I32TY]] [[#SPEC10]] 12
// CHECK-SPV-DAG: SpecConstant [[#I32TY]] [[#SPEC11]] 13
// CHECK-SPV-DAG: SpecConstant [[#I32TY]] [[#SPEC12]] 2
// CHECK-SPV-DAG: SpecConstantComposite [[#COMPTY]] {{.*}} [[#SPEC10]] [[#SPEC11]]
