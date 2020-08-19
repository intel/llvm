// RUN: %clang_cc1 -fsycl -fsycl-is-device -fsyntax-only -Wno-sycl-2017-compat -verify -DTRIGGER_ERROR %s
// RUN: %clang_cc1 -fsycl -fsycl-is-device -Wno-sycl-2017-compat -ast-dump %s | FileCheck %s

[[intel::reqd_sub_group_size(4)]] void foo() {} // expected-note {{conflicting attribute is here}}
// expected-note@-1 {{conflicting attribute is here}}
[[intel::reqd_sub_group_size(32)]] void baz() {} // expected-note {{conflicting attribute is here}}

class Functor16 {
public:
  // expected-warning@+2 {{attribute 'intel_reqd_sub_group_size' is deprecated}}
  // expected-note@+1 {{did you mean to use 'intel::reqd_sub_group_size' instead?}}
  [[cl::intel_reqd_sub_group_size(16)]] void operator()() const {}
};

class Functor8 { // expected-error {{conflicting attributes applied to a SYCL kernel}}
public:
  [[intel::reqd_sub_group_size(8)]] void operator()() const { // expected-note {{conflicting attribute is here}}
    foo();
  }
};

class Functor4 {
public:
  [[intel::reqd_sub_group_size(12)]] void operator()() const {}
};

class Functor {
public:
  void operator()() const {
    foo();
  }
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

void bar() {
  Functor16 f16;
  kernel<class kernel_name1>(f16);

  Functor f;
  kernel<class kernel_name2>(f);

#ifdef TRIGGER_ERROR
  Functor8 f8;
  kernel<class kernel_name3>(f8);

  kernel<class kernel_name4>([]() { // expected-error {{conflicting attributes applied to a SYCL kernel}}
    foo();
    baz();
  });
#endif

  kernel<class kernel_name5>([]() [[intel::reqd_sub_group_size(2)]]{});
  kernel<class kernel_name6>([]() [[intel::reqd_sub_group_size(4)]] { foo(); });
  // expected-warning@+2 {{attribute 'intel_reqd_sub_group_size' is deprecated}}
  // expected-note@+1 {{did you mean to use 'intel::reqd_sub_group_size' instead?}}
  kernel<class kernel_name7>([]() [[cl::intel_reqd_sub_group_size(6)]]{});

  Functor4 f4;
  kernel<class kernel_name8>(f4);
}

[[intel::reqd_sub_group_size(16)]] SYCL_EXTERNAL void B();
[[intel::reqd_sub_group_size(16)]] void A() {
}

[[intel::reqd_sub_group_size(16)]] SYCL_EXTERNAL void B() {
  A();
}

#ifdef TRIGGER_ERROR
// expected-note@+1 {{conflicting attribute is here}}
[[intel::reqd_sub_group_size(2)]] void sg_size2() {}

// expected-note@+2 {{conflicting attribute is here}}
// expected-error@+1 {{conflicting attributes applied to a SYCL kernel}}
[[intel::reqd_sub_group_size(4)]] __attribute__((sycl_device)) void sg_size4() {
  sg_size2();
}
#endif

// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name1
// CHECK: IntelReqdSubGroupSizeAttr {{.*}}
// CHECK-NEXT: IntegerLiteral{{.*}}16{{$}}
// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name2
// CHECK: IntelReqdSubGroupSizeAttr {{.*}}
// CHECK-NEXT: IntegerLiteral{{.*}}4{{$}}
// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name5
// CHECK: IntelReqdSubGroupSizeAttr {{.*}}
// CHECK-NEXT: IntegerLiteral{{.*}}2{{$}}
// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name7
// CHECK: IntelReqdSubGroupSizeAttr {{.*}}
// CHECK-NEXT: IntegerLiteral{{.*}}6{{$}}
// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name8
// CHECK: IntelReqdSubGroupSizeAttr {{.*}}
// CHECK-NEXT: IntegerLiteral{{.*}}12{{$}}
