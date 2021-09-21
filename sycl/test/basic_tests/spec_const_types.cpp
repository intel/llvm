// RUN: %clangxx -fsycl -fsycl-device-only -c -o %t.bc %s
// RUN: sycl-post-link %t.bc -spec-const=rt -o %t-split.txt
// RUN: cat %t-split_0.prop | FileCheck %s
// RUN: llvm-spirv -o %t-split_0.spv -spirv-max-version=1.1 -spirv-ext=+all %t-split_0.bc
//
//==----------- spec_const.cpp ---------------------------------------------==//
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

class SpecializedKernel;
class MyBoolConst;
class MyInt8Const;
class MyUInt8Const;
class MyInt16Const;
class MyUInt16Const;
class MyInt32Const;
class MyUInt32Const;
class MyInt64Const;
class MyUInt64Const;

class MyHalfConst;
class MyFloatConst;
class MyDoubleConst;

// Fetch a value at runtime.
int8_t get_value();

int main() {
  cl::sycl::queue queue;
  cl::sycl::program program(queue.get_context());

  // Create specialization constants.
  cl::sycl::ext::oneapi::experimental::spec_constant<bool, MyBoolConst> i1 =
      program.set_spec_constant<MyBoolConst>((bool)get_value());
  // CHECK-DAG: _ZTS11MyBoolConst=2|

  cl::sycl::ext::oneapi::experimental::spec_constant<int8_t, MyInt8Const> i8 =
      program.set_spec_constant<MyInt8Const>((int8_t)get_value());
  // CHECK-DAG: _ZTS11MyInt8Const=2|
  cl::sycl::ext::oneapi::experimental::spec_constant<uint8_t, MyUInt8Const>
      ui8 = program.set_spec_constant<MyUInt8Const>((uint8_t)get_value());
  // CHECK-DAG: _ZTS12MyUInt8Const=2|

  cl::sycl::ext::oneapi::experimental::spec_constant<int16_t, MyInt16Const>
      i16 = program.set_spec_constant<MyInt16Const>((int16_t)get_value());
  // CHECK-DAG: _ZTS12MyInt16Const=2|
  cl::sycl::ext::oneapi::experimental::spec_constant<uint16_t, MyUInt16Const>
      ui16 = program.set_spec_constant<MyUInt16Const>((uint16_t)get_value());
  // CHECK-DAG: _ZTS13MyUInt16Const=2|

  cl::sycl::ext::oneapi::experimental::spec_constant<int32_t, MyInt32Const>
      i32 = program.set_spec_constant<MyInt32Const>((int32_t)get_value());
  // CHECK-DAG: _ZTS12MyInt32Const=2|
  cl::sycl::ext::oneapi::experimental::spec_constant<uint32_t, MyUInt32Const>
      ui32 = program.set_spec_constant<MyUInt32Const>((uint32_t)get_value());
  // CHECK-DAG: _ZTS13MyUInt32Const=2|

  cl::sycl::ext::oneapi::experimental::spec_constant<int64_t, MyInt64Const>
      i64 = program.set_spec_constant<MyInt64Const>((int64_t)get_value());
  // CHECK-DAG: _ZTS12MyInt64Const=2|
  cl::sycl::ext::oneapi::experimental::spec_constant<uint64_t, MyUInt64Const>
      ui64 = program.set_spec_constant<MyUInt64Const>((uint64_t)get_value());
  // CHECK-DAG: _ZTS13MyUInt64Const=2|

#define HALF 0 // TODO not yet supported
#if HALF
  cl::sycl::ext::oneapi::experimental::spec_constant<cl::sycl::half,
                                                     MyHalfConst>
      f16 = program.set_spec_constant<MyHalfConst>((cl::sycl::half)get_value());
#endif

  cl::sycl::ext::oneapi::experimental::spec_constant<float, MyFloatConst> f32 =
      program.set_spec_constant<MyFloatConst>((float)get_value());
  // CHECK-DAG: _ZTS12MyFloatConst=2|

  cl::sycl::ext::oneapi::experimental::spec_constant<double, MyDoubleConst>
      f64 = program.set_spec_constant<MyDoubleConst>((double)get_value());
  // CHECK-DAG: _ZTS13MyDoubleConst=2|

  program.build_with_kernel_type<SpecializedKernel>();

  std::vector<float> vec(1);
  {
    cl::sycl::buffer<float, 1> buf(vec.data(), vec.size());

    queue.submit([&](cl::sycl::handler &cgh) {
      auto acc = buf.get_access<cl::sycl::access::mode::write>(cgh);
      cgh.single_task<SpecializedKernel>(
          program.get_kernel<SpecializedKernel>(), [=]() {
            acc[0] = i1.get() + i8.get() + ui8.get() + i16.get() + ui16.get() +
                     i32.get() + ui32.get() + i64.get() + ui64.get() +
#if HALF
                     f16.get() +
#endif
                     f32.get() + f64.get();
          });
    });
  }
}
