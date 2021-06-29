// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -Wno-return-type -Wno-sycl-2017-compat -fcxx-exceptions -fsyntax-only -ast-dump -verify -pedantic %s | FileCheck %s

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

struct FuncObj {
  [[intel::fpga_pipeline]] void operator()() const {}
};

int main() {
  q.submit([&](handler &h) {
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel1
    // CHECK:       SYCLIntelFpgaPipelineAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
    h.single_task<class test_kernel1>(FuncObj());

    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel2
    // CHECK:       SYCLIntelFpgaPipelineAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 0
    // CHECK-NEXT:  IntegerLiteral{{.*}}0{{$}}
    h.single_task<class test_kernel2>(
        []() [[intel::fpga_pipeline(0)]]{});

    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel3
    // CHECK: SYCLIntelFpgaPipelineAttr{{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
    h.single_task<class test_kernel3>(
        []() [[intel::fpga_pipeline(1)]]{});

    // Ignore duplicate attribute.
    h.single_task<class test_kernel4>(
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel4
    // CHECK:       SYCLIntelFpgaPipelineAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
        []() [[intel::fpga_pipeline,
               intel::fpga_pipeline]]{});

    // expected-error@+2{{integral constant expression must have integral or unscoped enumeration type, not 'const char [4]'}}
    h.single_task<class test_kernel5>(
        []() [[intel::fpga_pipeline("foo")]]{});

    h.single_task<class test_kernel6>([]() {
      // expected-error@+1{{'fpga_pipeline' attribute only applies to 'for', 'while', 'do' statements, and functions}}
      [[intel::fpga_pipeline(1)]] int a;
    });

    h.single_task<class test_kernel7>(
        []() [[intel::fpga_pipeline(0),      // expected-note {{previous attribute is here}}
	       intel::fpga_pipeline(1)]]{});  // expected-warning{{attribute 'fpga_pipeline' is already applied with different arguments}}
  });
  return 0;
}
