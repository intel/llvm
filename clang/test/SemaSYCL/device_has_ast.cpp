// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -ast-dump %s | FileCheck %s

// Tests for AST of device_has(aspect, ...) attribute
#include "sycl.hpp"

using namespace cl::sycl;
queue q;

// CHECK: FunctionDecl {{.*}} func1 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLDeviceHasAttr
// CHECK-NEXT: DeclRefExpr {{.*}} 'sycl::aspect' EnumConstant {{.*}} 'cpu' 'sycl::aspect'
[[sycl::device_has(cl::sycl::aspect::cpu)]] void func1() {}

// CHECK: FunctionDecl {{.*}} func2 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLDeviceHasAttr
// CHECK-NEXT: DeclRefExpr {{.*}} 'sycl::aspect' EnumConstant {{.*}} 'fp16' 'sycl::aspect'
// CHECK-NEXT: DeclRefExpr {{.*}} 'sycl::aspect' EnumConstant {{.*}} 'gpu' 'sycl::aspect'
[[sycl::device_has(cl::sycl::aspect::fp16, cl::sycl::aspect::gpu)]] void func2() {}

// CHECK: FunctionDecl {{.*}} func3 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLDeviceHasAttr
[[sycl::device_has()]] void func3() {}

// CHECK: FunctionDecl {{.*}} used func4 'void ()'
// CHECK-NEXT: TemplateArgument integral 0
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLDeviceHasAttr
// CHECK-NEXT: SubstNonTypeTemplateParmExpr {{.*}} 'sycl::aspect'
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}} referenced 'cl::sycl::aspect':'sycl::aspect' depth 0 index 0 Aspect
// CHECK-NEXT: CStyleCastExpr {{.*}} 'sycl::aspect' <IntegralCast>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 0
template <cl::sycl::aspect Aspect>
[[sycl::device_has(Aspect)]] void func4() {}

// CHECK: FunctionDecl {{.*}} used func5 'void ()'
// CHECK-NEXT: SYCLDeviceHasAttr
// CHECK-NEXT: DeclRefExpr {{.*}} 'sycl::aspect' EnumConstant {{.*}} 'cpu' 'sycl::aspect'
// CHECK-NEXT: FunctionDecl {{.*}} used func5 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLDeviceHasAttr {{.*}} Inherited
// CHECK-NEXT: DeclRefExpr {{.*}} 'sycl::aspect' EnumConstant {{.*}} 'cpu' 'sycl::aspect'
[[sycl::device_has(cl::sycl::aspect::cpu)]] void func5();
void func5() {}

// CHECK: FunctionDecl {{.*}} used func6 'void ()'
// CHECK-NEXT: SYCLDeviceHasAttr
// CHECK-NEXT: DeclRefExpr {{.*}} 'sycl::aspect' EnumConstant {{.*}} 'cpu' 'sycl::aspect'
[[sycl::device_has(cl::sycl::aspect::cpu)]] void func6();

// CHECK: FunctionDecl {{.*}} used func6 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLDeviceHasAttr
// CHECK-NEXT: DeclRefExpr {{.*}} 'sycl::aspect' EnumConstant {{.*}} 'gpu' 'sycl::aspect'
// CHECK-NOT: SYCLDeviceHasAttr
[[sycl::device_has(cl::sycl::aspect::gpu)]] void func6() {}

class KernelFunctor {
public:
  void operator()() const {
    func1();
    func2();
    func3();
    func4<cl::sycl::aspect::host>();
    func5();
    func6();
  }
};

void foo() {
  q.submit([&](handler &h) {
    KernelFunctor f1;
    h.single_task<class kernel_name_1>(f1);
  });
}
