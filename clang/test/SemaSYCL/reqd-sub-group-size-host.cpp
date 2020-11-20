// RUN: %clang_cc1 -fsycl -fsycl-is-host -Wno-sycl-2017-compat -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsycl -fsycl-is-host -Wno-sycl-2017-compat -ast-dump %s | FileCheck %s
//expected-no-diagnostics

[[intel::reqd_sub_group_size(8)]] void fun() {}

class Functor {
public:
  [[intel::reqd_sub_group_size(16)]] void operator()() const {}
};

class Functor16 {
public:
  [[intel::reqd_sub_group_size(4)]] void operator()() const {
  }
};

class Functor4 {
public:
  [[intel::reqd_sub_group_size(12)]] void operator()() const {}
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

  Functor4 f4;
  kernel<class kernel_name3>(f4);
}

// CHECK: FunctionDecl {{.*}} {{.*}} fun 'void ()'
// CHECK: IntelReqdSubGroupSizeAttr {{.*}} reqd_sub_group_size
// CHECK-NEXT: IntegerLiteral{{.*}} 'int' 8
// CHECK: CXXRecordDecl {{.*}} implicit class Functor
// CHECK: CXXMethodDecl {{.*}} operator() 'void () const'
// CHECK: IntelReqdSubGroupSizeAttr {{.*}} reqd_sub_group_size
// CHECK-NEXT: IntegerLiteral{{.*}} 'int' 16
// CHECK: CXXRecordDecl {{.*}} implicit class Functor16
// CHECK: CXXMethodDecl {{.*}} operator() 'void () const'
// CHECK: IntelReqdSubGroupSizeAttr {{.*}} reqd_sub_group_size
// CHECK-NEXT: IntegerLiteral{{.*}}'int' 4
// CXXRecordDecl {{.*}} implicit class Functor4
// CHECK: CXXMethodDecl {{.*}} operator() 'void () const'
// CHECK: IntelReqdSubGroupSizeAttr {{.*}} reqd_sub_group_size
// CHECK-NEXT: IntegerLiteral{{.*}} 'int' 12
