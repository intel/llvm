// RUN: %clang_cc1 -internal-isystem %S/Inputs -triple spir64-unknown-unknown \
// RUN:     -fsycl-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --implicit-check-not _ZN7Derived3baz \
// RUN:     --implicit-check-not _ZN4Base4baz --implicit-check-not _ZN4Base3foo
//
// Some SYCL properties may be turned into 'sycl_device' attribute implicitly
// and we would like to ensure that functions like this (at the moment those
// would be virtual member functions only) are forcefully emitted into device
// code.

class Base {
  virtual void foo() {}

  virtual void baz();

  [[__sycl_detail__::add_ir_attributes_function("indirectly-callable", "a")]]
  virtual void bar();
};

void Base::bar() {}

void Base::baz() {}

class Derived : public Base {
public:
  [[__sycl_detail__::add_ir_attributes_function("indirectly-callable", "b")]]
  void foo() override;

  [[__sycl_detail__::add_ir_attributes_function("indirectly-callable", "c")]]
  void bar() override final;

  [[__sycl_detail__::add_ir_attributes_function("not-indirectly-callable", "c")]]
  void baz() override final;
};

void Derived::foo() {}

void Derived::bar() {}

void Derived::baz() {}

// CHECK: define {{.*}}spir_func void @_ZN4Base3bar{{.*}} #[[#AttrA:]]
// CHECK: define {{.*}}spir_func void @_ZN7Derived3foo{{.*}} #[[#AttrB:]]
// CHECK: define {{.*}}spir_func void @_ZN7Derived3bar{{.*}} #[[#AttrC:]]
// CHECK: attributes #[[#AttrA]] = {{.*}} "indirectly-callable"="a"
// CHECK: attributes #[[#AttrB]] = {{.*}} "indirectly-callable"="b"
// CHECK: attributes #[[#AttrC]] = {{.*}} "indirectly-callable"="c"
