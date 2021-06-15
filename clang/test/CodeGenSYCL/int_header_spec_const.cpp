// RUN: %clang_cc1 -fsycl-is-device -fsycl-int-header=%t.h %s -o %t.out
// RUN: FileCheck -input-file=%t.h %s

#include "Inputs/sycl.hpp"

// This test verifies proper emission of specialization constants into the
// integration header.

class SpecializedKernel;
class MyBoolConst;
class MyInt8Const;
class MyUInt8Const;
class MyInt16Const;
class MyUInt16Const;
class MyInt32Const;
class MyUInt32Const;

class MyFloatConst;
class MyDoubleConst;

namespace test {
class MySpecConstantWithinANamespace;
};

int main() {
  // Create specialization constants.
  cl::sycl::ONEAPI::experimental::spec_constant<bool, MyBoolConst> i1(false);
  cl::sycl::ONEAPI::experimental::spec_constant<char, MyInt8Const> i8(0);
  cl::sycl::ONEAPI::experimental::spec_constant<unsigned char, MyUInt8Const> ui8(0);
  cl::sycl::ONEAPI::experimental::spec_constant<short, MyInt16Const> i16(0);
  cl::sycl::ONEAPI::experimental::spec_constant<unsigned short, MyUInt16Const> ui16(0);
  cl::sycl::ONEAPI::experimental::spec_constant<int, MyInt32Const> i32(0);
  // Constant used twice, but there must be single entry in the int header,
  // otherwise compilation error would be issued.
  cl::sycl::ONEAPI::experimental::spec_constant<int, MyInt32Const> i32_1(0);
  cl::sycl::ONEAPI::experimental::spec_constant<unsigned int, MyUInt32Const> ui32(0);
  cl::sycl::ONEAPI::experimental::spec_constant<float, MyFloatConst> f32(0);
  cl::sycl::ONEAPI::experimental::spec_constant<double, MyDoubleConst> f64(0);
  // Kernel name can be used as a spec constant name
  cl::sycl::ONEAPI::experimental::spec_constant<int, SpecializedKernel> spec1(0);
  // Spec constant name can be declared within a namespace
  cl::sycl::ONEAPI::experimental::spec_constant<int, test::MySpecConstantWithinANamespace> spec2(0);

  double val;
  double *ptr = &val; // to avoid "unused" warnings

  // CHECK: // Forward declarations of templated kernel function types:
  // CHECK: class SpecializedKernel;

  cl::sycl::kernel_single_task<SpecializedKernel>([=]() {
    *ptr = i1.get() +
           i8.get() +
           ui8.get() +
           i16.get() +
           ui16.get() +
           i32.get() +
           i32_1.get() +
           ui32.get() +
           f32.get() +
           f64.get() +
           spec1.get() +
           spec2.get();
  });
}
