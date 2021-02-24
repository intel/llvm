// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s
// Test code generation for sycl_device attribute.

int bar(int b);

int bar10(int a) { return a + 10; }
int bar20(int a) { return a + 20; }

class A {
public:
  // CHECK-DAG: define linkonce_odr spir_func void @_ZN1A3fooEv
  // CHECK-DAG: define {{.*}}spir_func i32 @_Z5bar20i
  __attribute__((sycl_device)) void foo() { bar20(10); }

  // CHECK-DAG: define linkonce_odr spir_func void @_ZN1AC1Ev
  // CHECK-DAG: define {{.*}}spir_func i32 @_Z5bar10i
  __attribute__((sycl_device))
  A() { bar10(10); }
  // CHECK-DAG: define linkonce_odr spir_func void @_ZN1AD1Ev
  __attribute__((sycl_device)) ~A() {}

  template <typename T>
  __attribute__((sycl_device)) void AFoo(T t) {}

  // Templates are emitted when they are instantiated
  // CHECK-DAG: define linkonce_odr spir_func void @_ZN1A4AFooIiEEvT_
  template <>
  __attribute__((sycl_device)) void AFoo<int>(int t) {}

  // CHECK-DAG: define linkonce_odr spir_func i32 @_ZN1A13non_annotatedEv
  int non_annotated() { return 1; }

  // CHECK-DAG: define linkonce_odr spir_func i32 @_ZN1A9annotatedEv
  __attribute__((sycl_device)) int annotated() { return non_annotated() + 1; }
};

template <typename T>
struct B {
  T data;
  B(T _data) : data(_data) {}

  __attribute__((sycl_device)) void BFoo(T t) {}
};

template <>
struct B<int> {
  int data;
  B(int _data) : data(_data) {}

  // CHECK-DAG: define linkonce_odr spir_func void @_ZN1BIiE4BFooEi
  __attribute__((sycl_device)) void BFoo(int t) {}
};

struct Base {
  // CHECK-DAG: define linkonce_odr spir_func void @_ZN4Base12BaseWithAttrEv
  __attribute__((sycl_device)) virtual void BaseWithAttr() { int a = 10; }
  virtual void BaseWithoutAttr() { int b = 20; }
};

struct Overrider : Base {
  // CHECK-DAG: define linkonce_odr spir_func void @_ZN9Overrider12BaseWithAttrEv
  __attribute__((sycl_device)) void BaseWithAttr() override { int a = 20; }
  // CHECK-DAG: define linkonce_odr spir_func void @_ZN9Overrider15BaseWithoutAttrEv
  __attribute__((sycl_device)) void BaseWithoutAttr() override { int b = 30; }
};

struct Overrider1 : Base {
  // CHECK-NOT: define linkonce_odr spir_func void @_ZN10Overrider112BaseWithAttrEv
  void BaseWithAttr() override { int a = 20; }
};

struct Finalizer : Base {
  // CHECK-DAG: define linkonce_odr spir_func void @_ZN9Finalizer12BaseWithAttrEv
  __attribute__((sycl_device)) void BaseWithAttr() final { int a = 20; }
  // CHECK-DAG: define linkonce_odr spir_func void @_ZN9Finalizer15BaseWithoutAttrEv
  __attribute__((sycl_device)) void BaseWithoutAttr() final { int b = 30; }
};

struct Finalizer1 : Base {
  // CHECK-NOT: define linkonce_odr spir_func void @_ZN10Finalizer112BaseWithAttrEv
  void BaseWithAttr() final { int a = 20; }
};

// CHECK-DAG: define {{.*}}spir_func i32 @_Z3fooii
__attribute__((sycl_device))
int foo(int a, int b) { return a + bar(b); }

// CHECK-DAG: define {{.*}}spir_func i32 @_Z3bari
int bar(int b) { return b; }

// CHECK-DAG: define {{.*}}spir_func i32 @_Z3fari
int far(int b) { return b; }

// CHECK-DAG: define {{.*}}spir_func i32 @_Z3booii
__attribute__((sycl_device))
int boo(int a, int b) { return a + far(b); }

// CHECK-DAG: define {{.*}}spir_func i32 @_Z3cari
__attribute__((sycl_device))
int car(int b);
int car(int b) { return b; }

// CHECK-DAG: define {{.*}}spir_func i32 @_Z3cazi
int caz(int b);
__attribute__((sycl_device))
int caz(int b) { return b; }

template<typename T>
__attribute__((sycl_device))
void taf(T t) {}

// CHECK-DAG: define weak_odr spir_func void @_Z3tafIiEvT_
template void taf<int>(int t);

// CHECK-DAG: define {{.*}}spir_func void @_Z3tafIcEvT_
template<> void taf<char>(char t) {}

template<typename T>
void tar(T t) {}

// CHECK-DAG: define {{.*}}spir_func void @_Z3tarIcEvT_
template<>
__attribute__((sycl_device))
void tar<char>(char t) {}

// CHECK-NOT: @_Z3tarIiEvT_
template void tar<int>(int t);

// CHECK-NOT: @_Z3gooi
int goo(int b) { return b; }
