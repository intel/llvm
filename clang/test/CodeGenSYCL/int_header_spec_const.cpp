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
  cl::sycl::ext::oneapi::experimental::spec_constant<bool, MyBoolConst> i1(false);
  cl::sycl::ext::oneapi::experimental::spec_constant<char, MyInt8Const> i8(0);
  cl::sycl::ext::oneapi::experimental::spec_constant<unsigned char, MyUInt8Const> ui8(0);
  cl::sycl::ext::oneapi::experimental::spec_constant<short, MyInt16Const> i16(0);
  cl::sycl::ext::oneapi::experimental::spec_constant<unsigned short, MyUInt16Const> ui16(0);
  cl::sycl::ext::oneapi::experimental::spec_constant<int, MyInt32Const> i32(0);
  // Constant used twice, but there must be single entry in the int header,
  // otherwise compilation error would be issued.
  cl::sycl::ext::oneapi::experimental::spec_constant<int, MyInt32Const> i32_1(0);
  cl::sycl::ext::oneapi::experimental::spec_constant<unsigned int, MyUInt32Const> ui32(0);
  cl::sycl::ext::oneapi::experimental::spec_constant<float, MyFloatConst> f32(0);
  cl::sycl::ext::oneapi::experimental::spec_constant<double, MyDoubleConst> f64(0);
  // Kernel name can be used as a spec constant name
  cl::sycl::ext::oneapi::experimental::spec_constant<int, SpecializedKernel> spec1(0);
  // Spec constant name can be declared within a namespace
  cl::sycl::ext::oneapi::experimental::spec_constant<int, test::MySpecConstantWithinANamespace> spec2(0);

  double val;
  double *ptr = &val; // to avoid "unused" warnings

  // CHECK: // Forward declarations of templated spec constant types:
  // CHECK: class MyInt8Const;
  // CHECK: class MyUInt8Const;
  // CHECK: class MyInt16Const;
  // CHECK: class MyUInt16Const;
  // CHECK: class MyInt32Const;
  // CHECK: class MyUInt32Const;
  // CHECK: class MyFloatConst;
  // CHECK: class MyDoubleConst;
  // CHECK: class SpecializedKernel;
  // CHECK: namespace test {
  // CHECK: class MySpecConstantWithinANamespace;
  // CHECK: }

  cl::sycl::kernel_single_task<SpecializedKernel>([=]() {
    *ptr = i1.get() +
           // CHECK-DAG: template <> struct sycl::detail::SpecConstantInfo<::MyBoolConst> {
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
           f64.get() +
           // CHECK-DAG: return "_ZTS13MyDoubleConst";
           spec1.get() +
           // CHECK-DAG: return "_ZTS17SpecializedKernel"
           spec2.get();
    // CHECK-DAG: return "_ZTSN4test30MySpecConstantWithinANamespaceE"
  });
}
