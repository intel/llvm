// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -Wno-return-type -Wno-sycl-2017-compat -fcxx-exceptions -fsyntax-only -sycl-std=2017 -ast-dump -DSYCL2017 -verify -pedantic %s 
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -Wno-return-type -Wno-sycl-2017-compat -fcxx-exceptions -fsyntax-only -sycl-std=2020 -ast-dump -DSYCL2020 -verify -pedantic %s

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

struct FuncObj {
  //expected-warning@+2 {{attribute 'intelfpga::no_global_work_offset' is deprecated}}
  //expected-note@+1 {{did you mean to use 'intel::no_global_work_offset' instead?}}
  [[intelfpga::no_global_work_offset]] void operator()() const {}
};

[[intel::no_global_work_offset(1)]] void func() {}

int main() {
  q.submit([&](handler &h) {
    // CHECK:       SYCLIntelNoGlobalWorkOffsetAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
    h.single_task<class test_kernel1>(FuncObj());

    // CHECK: SYCLIntelNoGlobalWorkOffsetAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 0
    // CHECK-NEXT:  IntegerLiteral{{.*}}0{{$}}
    h.single_task<class test_kernel2>(
        []() [[intel::no_global_work_offset(0)]]{});

    // CHECK: SYCLIntelNoGlobalWorkOffsetAttr{{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 42
    // CHECK-NEXT:  IntegerLiteral{{.*}}42{{$}}
    h.single_task<class test_kernel3>(
        []() [[intel::no_global_work_offset(42)]]{});

    // CHECK: SYCLIntelNoGlobalWorkOffsetAttr{{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int -1
    // CHECK-NEXT: UnaryOperator{{.*}} 'int' prefix '-'
    // CHECK-NEXT-NEXT: IntegerLiteral{{.*}}1{{$}}
    h.single_task<class test_kernel4>(
        []() [[intel::no_global_work_offset(-1)]]{});

    // Ignore duplicate attribute.
    h.single_task<class test_kernel5>(
    // CHECK:       SYCLIntelNoGlobalWorkOffsetAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
        []() [[intel::no_global_work_offset,
               intel::no_global_work_offset]]{}); // OK

    // expected-error@+2{{integral constant expression must have integral or unscoped enumeration type, not 'const char [4]'}}
    h.single_task<class test_kernel6>(
        []() [[intel::no_global_work_offset("foo")]]{});

    h.single_task<class test_kernel7>([]() {
      // expected-error@+1{{'no_global_work_offset' attribute only applies to functions}}
      [[intel::no_global_work_offset(1)]] int a;
    });

    h.single_task<class test_kernel8>(
        []() [[intel::no_global_work_offset(0),      // expected-note {{previous attribute is here}}
	       intel::no_global_work_offset(1)]]{});  // expected-warning{{attribute 'no_global_work_offset' is already applied with different arguments}}

#if defined(SYCL2020)
    // Test attribute is not propagated.
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel9
    // CHECK-NOT:   SYCLIntelNoGlobalWorkOffsetAttr {{.*}}
    h.single_task<class test_kernel9>(
        []() { func(); });
#endif // SYCL2020

#if defined(SYCL2017)
    // Test attribute is propagated.
    // CHECK-LABEL: FunctionDecl {{.*}}test_kerne10
    // CHECK:       SYCLIntelNoGlobalWorkOffsetAttr {{.*}}
    // CHECK-NEXT:  ConstantExpr {{.*}} 'int'
    // CHECK-NEXT:  value: Int 1
    // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
    h.single_task<class test_kernel10>(
        []() { func(); });
#endif // SYCL2017
  });
  return 0;
}
