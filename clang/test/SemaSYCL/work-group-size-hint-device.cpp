// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2017 -Wno-sycl-2017-compat -ast-dump %s -DEXPECT_PROP | FileCheck %s
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -ast-dump %s | FileCheck %s

#include "sycl.hpp"

// Show that the attribute works on member functions.
class Functor_1 {
public:
  [[sycl::work_group_size_hint(16, 2)]] void operator()() const {};
};

// CHECK: CXXRecordDecl {{.*}} {{.*}}Functor_1
// CHECK: WorkGroupSizeHintAttr {{.*}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 16
// CHECK-NEXT:  IntegerLiteral{{.*}}16{{$}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 2
// CHECK-NEXT:  IntegerLiteral{{.*}}2{{$}}
// CHECK-NEXT:  ConstantExpr{{.*}}'int'
// CHECK-NEXT:  value: Int 1
// CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}

[[sycl::work_group_size_hint(4, 4, 4)]] void f4x4x4(){};

void invoke() {
  Functor_1 f1;
  sycl::queue q;

  q.submit([&](sycl::handler &h) {
    h.single_task<class kernel_name1>(f1);
    // CHECK: FunctionDecl {{.*}} {{.*}}kernel_name1
    // CHECK: WorkGroupSizeHintAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 16
    // CHECK-NEXT:  IntegerLiteral{{.*}}16{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 2
    // CHECK-NEXT:  IntegerLiteral{{.*}}2{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}

    // Checking that attributes are propagated to the kernel from functions in SYCL 1.2.1 mode.
#ifdef EXPECT_PROP
    h.single_task<class kernel_name2>([=]() {
      f4x4x4();
    });
#else
    // Manually specifying the attribute.
    h.single_task<class kernel_name2>([=] [[sycl::work_group_size_hint(4, 4, 4)]] () {});
#endif
    // CHECK: FunctionDecl {{.*}} {{.*}}kernel_name2
    // CHECK: WorkGroupSizeHintAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}

    h.single_task<class kernel_name3>([=] [[sycl::work_group_size_hint(2, 4)]] () {
      f1(); // There should be no conflict between hints
    });
    // CHECK: FunctionDecl {{.*}} {{.*}}kernel_name3
    // CHECK: WorkGroupSizeHintAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 2
    // CHECK-NEXT:  IntegerLiteral{{.*}}2{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 4
    // CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}
    // CHECK-NEXT:  ConstantExpr{{.*}}'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
  });
}
