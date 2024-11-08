// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -ast-dump %s | FileCheck %s

// Tests for AST of device_has(aspect, ...) attribute
#include "sycl.hpp"

using namespace sycl;
queue q;

// CHECK: FunctionDecl {{.*}} func1 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLDeviceHasAttr
// CHECK-NEXT: DeclRefExpr {{.*}} 'sycl::aspect' EnumConstant {{.*}} 'cpu' 'sycl::aspect'
[[sycl::device_has(sycl::aspect::cpu)]] void func1() {}

// CHECK: FunctionDecl {{.*}} func2 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLDeviceHasAttr
// CHECK-NEXT: DeclRefExpr {{.*}} 'sycl::aspect' EnumConstant {{.*}} 'fp16' 'sycl::aspect'
// CHECK-NEXT: NestedNameSpecifier TypeSpec 'sycl::aspect'
// CHECK-NEXT: NestedNameSpecifier Namespace {{.*}} 'sycl'
// CHECK-NEXT: DeclRefExpr {{.*}} 'sycl::aspect' EnumConstant {{.*}} 'gpu' 'sycl::aspect'
// CHECK-NEXT: NestedNameSpecifier TypeSpec 'sycl::aspect'
// CHECK-NEXT: NestedNameSpecifier Namespace {{.*}} 'sycl'
[[sycl::device_has(sycl::aspect::fp16, sycl::aspect::gpu)]] void func2() {}

// CHECK: FunctionDecl {{.*}} func3 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLDeviceHasAttr
[[sycl::device_has()]] void func3() {}

// CHECK: FunctionDecl {{.*}} used func4 'void ()'
// CHECK-NEXT: TemplateArgument integral 'sycl::aspect::host'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLDeviceHasAttr
// CHECK-NEXT: SubstNonTypeTemplateParmExpr {{.*}} 'sycl::aspect'
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}} referenced 'sycl::aspect' depth 0 index 0 Aspect
// CHECK-NEXT: CStyleCastExpr {{.*}} 'sycl::aspect' <IntegralCast>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 0
template <sycl::aspect Aspect>
[[sycl::device_has(Aspect)]] void func4() {}

// CHECK: FunctionDecl {{.*}} used func5 'void ()'
// CHECK-NEXT: SYCLDeviceHasAttr
// CHECK-NEXT: DeclRefExpr {{.*}} 'sycl::aspect' EnumConstant {{.*}} 'cpu' 'sycl::aspect'
// CHECK-NEXT: NestedNameSpecifier TypeSpec 'sycl::aspect'
// CHECK-NEXT: NestedNameSpecifier Namespace {{.*}} 'sycl'
// CHECK-NEXT: FunctionDecl {{.*}} used func5 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLDeviceHasAttr {{.*}} Inherited
// CHECK-NEXT: DeclRefExpr {{.*}} 'sycl::aspect' EnumConstant {{.*}} 'cpu' 'sycl::aspect'
// CHECK-NEXT: NestedNameSpecifier TypeSpec 'sycl::aspect'
// CHECK-NEXT: NestedNameSpecifier Namespace {{.*}} 'sycl'
[[sycl::device_has(sycl::aspect::cpu)]] void func5();
void func5() {}

// CHECK: FunctionDecl {{.*}} used func6 'void ()'
// CHECK-NEXT: SYCLDeviceHasAttr
// CHECK-NEXT: DeclRefExpr {{.*}} 'sycl::aspect' EnumConstant {{.*}} 'cpu' 'sycl::aspect'
[[sycl::device_has(sycl::aspect::cpu)]] void func6();

// CHECK: FunctionDecl {{.*}} used func6 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLDeviceHasAttr
// CHECK-NEXT: DeclRefExpr {{.*}} 'sycl::aspect' EnumConstant {{.*}} 'gpu' 'sycl::aspect'
// CHECK-NOT: SYCLDeviceHasAttr
[[sycl::device_has(sycl::aspect::gpu)]] void func6() {}

// CHECK: FunctionTemplateDecl {{.*}} func7
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}} ... Asp
// CHECK-NEXT: FunctionDecl {{.*}} func7
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLDeviceHasAttr
// CHECK-NEXT: PackExpansionExpr
// CHECK-NEXT: DeclRefExpr {{.*}} NonTypeTemplateParm {{.*}} 'Asp' 'sycl::aspect'

// CHECK: FunctionDecl {{.*}} func7 'void ()'
// CHECK-NEXT: TemplateArgument pack
// CHECK-NEXT: TemplateArgument integral 'sycl::aspect::cpu'
// CHECK-NEXT: TemplateArgument integral 'sycl::aspect::gpu'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLDeviceHasAttr
// CHECK-NEXT: SubstNonTypeTemplateParmExpr {{.*}} 'sycl::aspect'
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}} ... Asp
// CHECK-NEXT: CStyleCastExpr {{.*}} 'sycl::aspect' <IntegralCast>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT: SubstNonTypeTemplateParmExpr {{.*}} 'sycl::aspect'
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}} ... Asp
// CHECK-NEXT: CStyleCastExpr {{.*}} 'sycl::aspect' <IntegralCast>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 2

// CHECK: FunctionDecl {{.*}} func7 'void ()'
// CHECK-NEXT: TemplateArgument pack
// CHECK-NEXT: TemplateArgument integral 'sycl::aspect::cpu'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLDeviceHasAttr
// CHECK-NEXT: SubstNonTypeTemplateParmExpr {{.*}} 'sycl::aspect'
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}} ... Asp
// CHECK-NEXT: CStyleCastExpr {{.*}} 'sycl::aspect' <IntegralCast>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
template <sycl::aspect... Asp>
[[sycl::device_has(Asp...)]] void func7() {}

// CHECK: FunctionTemplateDecl {{.*}} func8
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}} Asp
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}} ... AspPack
// CHECK-NEXT: FunctionDecl {{.*}} func8
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLDeviceHasAttr
// CHECK-NEXT: DeclRefExpr {{.*}} NonTypeTemplateParm {{.*}} 'Asp' 'sycl::aspect'
// CHECK-NEXT: PackExpansionExpr
// CHECK-NEXT: DeclRefExpr {{.*}} NonTypeTemplateParm {{.*}} 'AspPack' 'sycl::aspect'

// CHECK-NEXT: FunctionDecl {{.*}} func8
// CHECK-NEXT: TemplateArgument integral 'sycl::aspect::host'
// CHECK-NEXT: TemplateArgument pack
// CHECK-NEXT: TemplateArgument integral 'sycl::aspect::cpu'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: SYCLDeviceHasAttr
// CHECK-NEXT: SubstNonTypeTemplateParmExpr {{.*}} 'sycl::aspect'
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}} Asp
// CHECK-NEXT: CStyleCastExpr {{.*}} 'sycl::aspect' <IntegralCast>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT: SubstNonTypeTemplateParmExpr {{.*}} 'sycl::aspect'
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}} ... AspPack
// CHECK-NEXT: CStyleCastExpr {{.*}} 'sycl::aspect' <IntegralCast>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
template <sycl::aspect Asp, sycl::aspect... AspPack>
[[sycl::device_has(Asp, AspPack...)]] void func8() {}

// CHECK: CXXRecordDecl {{.*}} KernelFunctor definition
class KernelFunctor {
public:
  void operator()() const {
    func1();
    func2();
    func3();
    func4<sycl::aspect::host>();
    func5();
    func6();
    func7<sycl::aspect::cpu, sycl::aspect::gpu>();
    func7<sycl::aspect::cpu>();
    func8<sycl::aspect::host, sycl::aspect::cpu>();
  }
};

// CHECK: CXXRecordDecl {{.*}} KernelFunctorAttr
class KernelFunctorAttr {
public:
  // CHECK: CXXMethodDecl {{.*}} used operator() 'void () const'
  // CHECK: SYCLDeviceHasAtt
  // CHECK-NEXT: DeclRefExpr {{.*}} 'sycl::aspect' EnumConstant {{.*}} 'cpu' 'sycl::aspect'
  [[sycl::device_has(sycl::aspect::cpu)]] void operator()() const {}
};

void foo() {
  q.submit([&](handler &h) {
    // Attributes applied to functions called from the kernel should not be propagated to kernel.
    // CHECK: FunctionDecl {{.*}}kernel_name_1
    // CHECK-NOT: SYCLDeviceHasAttr
    KernelFunctor f1;
    h.single_task<class kernel_name_1>(f1);

    // Attribute applied to operator() method of kernel functor is applied to kernel function
    // CHECK: FunctionDecl {{.*}}kernel_name_2
    // CHECK: SYCLDeviceHasAttr
    // CHECK-NEXT: DeclRefExpr {{.*}} 'sycl::aspect' EnumConstant {{.*}} 'cpu' 'sycl::aspect'
    KernelFunctorAttr f2;
    h.single_task<class kernel_name_2>(f2);

    // Attribute applied to kernel lambda
    // CHECK: FunctionDecl {{.*}}kernel_name_3
    // CHECK: SYCLDeviceHasAttr
    // CHECK-NEXT: DeclRefExpr {{.*}} 'sycl::aspect' EnumConstant {{.*}} 'gpu' 'sycl::aspect'
    h.single_task<class kernel_name_3>([]() [[sycl::device_has(sycl::aspect::gpu)]] {});
  });
}
