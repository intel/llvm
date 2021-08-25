// RUN: %clangxx -fsycl -fsycl-device-only -c -o %t.bc %s
// RUN: sycl-post-link %t.bc -spec-const=rt -o %t-split1.txt
// RUN: cat %t-split1_0.prop | FileCheck %s -check-prefixes=CHECK,CHECK-RT
// RUN: sycl-post-link %t.bc -spec-const=default -o %t-split2.txt
// RUN: cat %t-split2_0.prop | FileCheck %s -check-prefixes=CHECK,CHECK-DEF
// RUN: llvm-spirv -o %t-split1_0.spv -spirv-max-version=1.1 -spirv-ext=+all %t-split1_0.bc
// RUN: llvm-spirv -o %t-split2_0.spv -spirv-max-version=1.1 -spirv-ext=+all %t-split2_0.bc
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

#include <CL/sycl.hpp>

#include <cstdint>
#include <vector>

// FIXME: there is a bug with generation SPIR-V friendly IR for boolean spec
// constants, which was likely interoduced by https://github.com/intel/llvm/pull/3513
// constexpr sycl::specialization_id<bool> bool_id;
constexpr sycl::specialization_id<int8_t> int8_id(1);
constexpr sycl::specialization_id<uint8_t> uint8_id(2);
constexpr sycl::specialization_id<int16_t> int16_id(3);
constexpr sycl::specialization_id<uint16_t> uint16_id(4);
constexpr sycl::specialization_id<int32_t> int32_id(5);
constexpr sycl::specialization_id<uint32_t> uint32_id(6);
constexpr sycl::specialization_id<int64_t> int64_id(7);
constexpr sycl::specialization_id<uint64_t> uint64_id(8);
// FIXME: enable once we support constexpr constructor for half
// constexpr sycl::specialization_id<half> half_id(9.0);
constexpr sycl::specialization_id<float> float_id(10.0);
constexpr sycl::specialization_id<double> double_id(11.0);
constexpr sycl::marray<double, 5> ma;
constexpr sycl::specialization_id<sycl::marray<double, 5>> marray_id5(11.0);
constexpr sycl::specialization_id<sycl::marray<double, 1>> marray_id1(11.0);
constexpr sycl::specialization_id<sycl::marray<double, 5>> marray_id_def(ma);
constexpr sycl::vec<double, 4> v{};
constexpr sycl::specialization_id<sycl::vec<double, 4>> vec_id_def(v);
constexpr sycl::specialization_id<sycl::vec<double, 1>> vec_id1(11.0);
constexpr sycl::specialization_id<sycl::vec<double, 4>> vec_id4(11.0);

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

        // see FIXMEs above about bool and half types support
        acc[0] = /*i1 +*/ i8 + u8 + i16 + u16 + i32 + u32 + i64 + u64 +
                 //f16.get() +
                 f32 + f64 + c.a + c.b;
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
// FIXME: check line for half constant

// CHECK-RT-NOT: [SYCL/specialization constants default values]
// CHECK-DEF: [SYCL/specialization constants default values]
// CHECK-DEF: all=2|
