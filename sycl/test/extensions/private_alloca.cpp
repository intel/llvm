// RUN: %clangxx -fsycl -fsycl-device-only -c -o %t.bc %s
// RUN: %if asserts %{sycl-post-link -properties -debug-only=SpecConst %t.bc -spec-const=native -o %t.txt 2>&1 | FileCheck %s -check-prefixes=CHECK-LOG %} %else %{sycl-post-link %t.bc -properties -spec-const=native -o %t.txt 2>&1 %}
// RUN: cat %t_0.prop | FileCheck %s -check-prefixes=CHECK,CHECK-RT
// RUN: llvm-spirv -o %t_0.spv -spirv-max-version=1.1 -spirv-ext=+all %t_0.bc
// RUN: llvm-spirv -o - --to-text %t_0.spv | FileCheck %s -check-prefixes=CHECK-SPV

// Check SPIR-V code generation for 'sycl_ext_oneapi_private_alloca'. Each call
// to the extension API is annotated as follows for future reference:
//
// <NAME>: element_type=<et>, alignment=<align>
//
// - <NAME>: Variable name in the test below. These will be the result of
// bitcasting a variable to a different pointer type. We use this instead of the
// variable due to FileCheck limitations.
// - <et>: element type. 'Bitcast X <NAME> Y' will originate value <NAME>, being
// X a pointer to <et> and storage class function.
// - <align>: alignment. <NAME> will appear in a 'Decorage <NAME> Aligment
// <align>' operation.

#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/experimental/alloca.hpp>
#include <sycl/specialization_id.hpp>

enum class enumeration { a, b, c };

struct composite {
  int a;
  int b;
  composite() = default;
};

constexpr sycl::specialization_id<int8_t> int8_id(42);
constexpr sycl::specialization_id<int16_t> int16_id(34);
constexpr sycl::specialization_id<uint32_t> uint32_id(46);
constexpr sycl::specialization_id<int32_t> int32_id(52);
constexpr sycl::specialization_id<uint64_t> uint64_id(81);

template <typename... Ts> SYCL_EXTERNAL void keep(const Ts &...);

SYCL_EXTERNAL void test(sycl::kernel_handler &kh) {
  keep(/*B0: storage_class=function, element_type=f32, alignment=4*/
       sycl::ext::oneapi::experimental::private_alloca<
           float, int8_id, sycl::access::decorated::yes>(kh),
       /*B1: element_type=f64, alignment=8*/
       sycl::ext::oneapi::experimental::private_alloca<
           double, uint32_id, sycl::access::decorated::no>(kh),
       /*B2: element_type=i32, alignment=4*/
       sycl::ext::oneapi::experimental::private_alloca<
           int, int16_id, sycl::access::decorated::legacy>(kh),
       /*B3: element_type=i64, alignment=16*/
       sycl::ext::oneapi::experimental::aligned_private_alloca<
           int64_t, alignof(int64_t) * 2, uint64_id,
           sycl::access::decorated::no>(kh),
       /*B4: element_type=composite, alignment=32*/
       sycl::ext::oneapi::experimental::aligned_private_alloca<
           composite, alignof(composite) * 8, int32_id,
           sycl::access::decorated::yes>(kh));
}

// CHECK: [SYCL/specialization constants]
// CHECK-DAG: [[UNIQUE_PREFIX:[a-z0-9]+]]____ZL7int8_id=2|
// CHECK-DAG: [[UNIQUE_PREFIX]]____ZL8int16_id=2|
// CHECK-DAG: [[UNIQUE_PREFIX]]____ZL8int32_id=2|
// CHECK-DAG: [[UNIQUE_PREFIX]]____ZL9uint32_id=2|
// CHECK-DAG: [[UNIQUE_PREFIX]]____ZL9uint64_id=2|

// CHECK-RT: [SYCL/specialization constants default values]
// CHECK-DEF: [SYCL/specialization constants default values]
// CHECK-DEF: all=2|

// CHECK-LOG: sycl.specialization-constants
// CHECK-LOG:[[UNIQUE_PREFIX:[a-z0-9]+]]____ZL7int8_id={0, 0, 1}
// CHECK-NEXT-LOG:[[UNIQUE_PREFIX]]____ZL8int16_id={2, 0, 2}
// CHECK-NEXT-LOG:[[UNIQUE_PREFIX]]____ZL8int32_id={4, 0, 4}
// CHECK-NEXT-LOG:[[UNIQUE_PREFIX]]____ZL9uint32_id={5, 0, 4}
// CHECK-NEXT-LOG:[[UNIQUE_PREFIX]]____ZL8int64_id={6, 0, 8}
// CHECK-NEXT-LOG:{0, 1, 42}
// CHECK-NEXT-LOG:{2, 2, 34}
// CHECK-NEXT-LOG:{6, 4, 52}
// CHECK-NEXT-LOG:{10, 4, 46}
// CHECK-NEXT-LOG:{22, 8, 81}

// CHECK-SPV-DAG: Decorate [[#SPEC0:]] SpecId 0
// CHECK-SPV-DAG: Decorate [[#SPEC1:]] SpecId 1
// CHECK-SPV-DAG: Decorate [[#SPEC2:]] SpecId 2
// CHECK-SPV-DAG: Decorate [[#SPEC3:]] SpecId 3
// CHECK-SPV-DAG: Decorate [[#SPEC4:]] SpecId 4
// CHECK-SPV-DAG: TypeInt [[#I8TY:]]  8 0
// CHECK-SPV-DAG: TypeInt [[#I16TY:]] 16 0
// CHECK-SPV-DAG: TypeInt [[#I32TY:]] 32 0
// CHECK-SPV-DAG: TypeInt [[#I64TY:]] 64 0
// CHECK-SPV-DAG: TypeFloat [[#F32TY:]] 32
// CHECK-SPV-DAG: TypeFloat [[#F64TY:]] 64
// CHECK-SPV-DAG: TypeStruct [[#COMPTY:]] [[#I32TY]] [[#I32TY]]
// CHECK-SPV-DAG: SpecConstant [[#I8TY]]  [[#SPEC0]] 42
// CHECK-SPV-DAG: SpecConstant [[#I32TY]] [[#SPEC1]] 46
// CHECK-SPV-DAG: SpecConstant [[#I16TY]] [[#SPEC2]] 34
// CHECK-SPV-DAG: SpecConstant [[#I64TY]] [[#SPEC3]] 81
// CHECK-SPV-DAG: SpecConstant [[#I32TY]] [[#SPEC4]] 52
// CHECK-SPV-DAG: TypeArray [[#ARRF32TY:]] [[#F32TY]] [[#SPEC0]]
// CHECK-SPV-DAG: TypePointer [[#ARRF32PTRTY:]] [[#FUNCTIONSTORAGE:]] [[#ARRF32TY]]
// CHECK-SPV-DAG: TypePointer [[#F32PTRTY:]] [[#FUNCTIONSTORAGE]] [[#F32TY]]
// CHECK-SPV-DAG: TypeArray [[#ARRF64TY:]] [[#F64TY]] [[#SPEC1]]
// CHECK-SPV-DAG: TypePointer [[#ARRF64PTRTY:]] [[#FUNCTIONSTORAGE]] [[#ARRF64TY]]
// CHECK-SPV-DAG: TypePointer [[#F64PTRTY:]] [[#FUNCTIONSTORAGE]] [[#F64TY]]
// CHECK-SPV-DAG: TypeArray [[#ARRI32TY:]] [[#I32TY]] [[#SPEC2]]
// CHECK-SPV-DAG: TypePointer [[#ARRI32PTRTY:]] [[#FUNCTIONSTORAGE]] [[#ARRI32TY]]
// CHECK-SPV-DAG: TypePointer [[#I32PTRTY:]] [[#FUNCTIONSTORAGE]] [[#I32TY]]
// CHECK-SPV-DAG: TypeArray [[#ARRI64TY:]] [[#I64TY]] [[#SPEC3]]
// CHECK-SPV-DAG: TypePointer [[#ARRI64PTRTY:]] [[#FUNCTIONSTORAGE]] [[#ARRI64TY]]
// CHECK-SPV-DAG: TypePointer [[#I64PTRTY:]] [[#FUNCTIONSTORAGE]] [[#I64TY]]
// CHECK-SPV-DAG: TypeArray [[#ARRCOMPTY:]] [[#COMPTY]] [[#SPEC4]]
// CHECK-SPV-DAG: TypePointer [[#ARRCOMPPTRTY:]] [[#FUNCTIONSTORAGE]] [[#ARRCOMPTY]]
// CHECK-SPV-DAG: TypePointer [[#COMPPTRTY:]] [[#FUNCTIONSTORAGE]] [[#COMPTY]]
// CHECK-SPV-DAG: Variable [[#ARRF32PTRTY]] [[#V0:]] [[#FUNCTIONSTORAGE]]
// CHECK-SPV-DAG: Bitcast [[#F32PTRTY]] [[#B0:]] [[#V0]]
// CHECK-SPV-DAG: Store {{.*}} [[#B0]]
// CHECK-SPV-DAG: Variable [[#ARRF64PTRTY]] [[#V1:]] [[#FUNCTIONSTORAGE]]
// CHECK-SPV-DAG: Bitcast [[#F64PTRTY]] [[#B1:]] [[#V1]]
// CHECK-SPV-DAG: Store {{.*}} [[#B1]]
// CHECK-SPV-DAG: Variable [[#ARRI32PTRTY]] [[#V2:]] [[#FUNCTIONSTORAGE]]
// CHECK-SPV-DAG: Bitcast [[#I32PTRTY]] [[#B2:]] [[#V2]]
// CHECK-SPV-DAG: Store {{.*}} [[#B2]]
// CHECK-SPV-DAG: Variable [[#ARRI64PTRTY]] [[#V3:]] [[#FUNCTIONSTORAGE]]
// CHECK-SPV-DAG: Bitcast [[#I64PTRTY]] [[#B3:]] [[#V3]]
// CHECK-SPV-DAG: Store {{.*}} [[#B3]]
// CHECK-SPV-DAG: Variable [[#ARRCOMPPTRTY]] [[#V4:]] [[#FUNCTIONSTORAGE]]
// CHECK-SPV-DAG: Bitcast [[#COMPPTRTY]] [[#B4:]] [[#V4]]
// CHECK-SPV-DAG: Store {{.*}} [[#B4]]
// CHECK-SPV-DAG: Decorate [[#B0]] Alignment 4
// CHECK-SPV-DAG: Decorate [[#B1]] Alignment 8
// CHECK-SPV-DAG: Decorate [[#B2]] Alignment 4
// CHECK-SPV-DAG: Decorate [[#B3]] Alignment 16
// CHECK-SPV-DAG: Decorate [[#B4]] Alignment 32
