// RUN: %clang_cc1 -I %S/Inputs -fsycl -fsycl-is-device -fsycl-int-header=%t.h %s -o %t.out
// RUN: FileCheck -input-file=%t.h %s

#include "sycl.hpp"

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

int main() {
  // Create specialization constants.
  cl::sycl::experimental::spec_constant<bool, MyBoolConst> i1(false);
  cl::sycl::experimental::spec_constant<char, MyInt8Const> i8(0);
  cl::sycl::experimental::spec_constant<unsigned char, MyUInt8Const> ui8(0);
  cl::sycl::experimental::spec_constant<short, MyInt16Const> i16(0);
  cl::sycl::experimental::spec_constant<unsigned short, MyUInt16Const> ui16(0);
  cl::sycl::experimental::spec_constant<int, MyInt32Const> i32(0);
  // Constant used twice, but there must be single entry in the int header,
  // otherwise compilation error would be issued.
  cl::sycl::experimental::spec_constant<int, MyInt32Const> i32_1(0);
  cl::sycl::experimental::spec_constant<unsigned int, MyUInt32Const> ui32(0);
  cl::sycl::experimental::spec_constant<float, MyFloatConst> f32(0);
  cl::sycl::experimental::spec_constant<double, MyDoubleConst> f64(0);

  double val;
  double *ptr = &val; // to avoid "unused" warnings

  cl::sycl::kernel_single_task<SpecializedKernel>([=]() {
    *ptr = i1.get() +
           // CHECK-DAG: template <> struct sycl::detail::SpecConstantInfo<class MyBoolConst> {
           // CHECK-DAG-NEXT:   static constexpr const char* getName() {
           // CHECK-DAG-NEXT:     return "_ZTS11MyBoolConst";
           // CHECK-DAG-NEXT:   }
           // CHECK-DAG-NEXT: };
           i8.get() +
           // CHECK-DAG: return "_ZTS11MyInt8Const";
           ui8.get() +
           // CHECK-DAG: return "_ZTS12MyUInt8Const";
           i16.get() +
           // CHECK-DAG: return "_ZTS12MyInt16Const";
           ui16.get() +
           // CHECK-DAG: return "_ZTS13MyUInt16Const";
           i32.get() +
           i32_1.get() +
           // CHECK-DAG: return "_ZTS12MyInt32Const";
           ui32.get() +
           // CHECK-DAG: return "_ZTS13MyUInt32Const";
           f32.get() +
           // CHECK-DAG: return "_ZTS12MyFloatConst";
           f64.get();
    // CHECK-DAG: return "_ZTS13MyDoubleConst";
  });
}
