// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

void helper() {}

[[intel::device_indirectly_callable]]
void foo() {
  helper();
}

// CHECK: define {{.*}}spir_func void @{{.*foo.*}}() #[[ATTRS_INDIR_CALL:[0-9]+]]
// CHECK: call spir_func void @{{.*helper.*}}()
//
// CHECK: define {{.*}}spir_func void @{{.*helper.*}}() #[[ATTRS_NOT_INDIR_CALL:[0-9]+]]
//

int bar20(int a) { return a + 20; }

class A {
public:
  // CHECK-DAG: define linkonce_odr spir_func void @_ZN1A3fooEv{{.*}}#[[ATTRS_INDIR_CALL]]
  // CHECK-DAG: define {{.*}}spir_func i32 @_Z5bar20{{.*}}#[[ATTRS_NOT_INDIR_CALL]]
  [[intel::device_indirectly_callable]] void foo() { bar20(10); }

  // CHECK-DAG: define linkonce_odr spir_func void @_ZN1AC1Ev{{.*}}#[[ATTRS_INDIR_CALL_1:[0-9]+]]
  [[intel::device_indirectly_callable]] A() {}
  // CHECK-DAG: define linkonce_odr spir_func void @_ZN1AD1Ev{{.*}}#[[ATTRS_INDIR_CALL_1]]
  [[intel::device_indirectly_callable]] ~A() {}

  template <typename T>
  [[intel::device_indirectly_callable]] void AFoo(T t) {}

  // Templates are emitted when they are instantiated
  // CHECK-DAG: define linkonce_odr spir_func void @_ZN1A4AFooIiEEvT_{{.*}}#[[ATTRS_INDIR_CALL]]
  template <>
  [[intel::device_indirectly_callable]] void AFoo<int>(int t) {}
};

struct Base {
  // CHECK-DAG: define linkonce_odr spir_func void @_ZN4Base12BaseWithAttrEv{{.*}}#[[ATTRS_INDIR_CALL]]
  [[intel::device_indirectly_callable]] virtual void BaseWithAttr() { int a = 10; }
  virtual void BaseWithoutAttr() { int b = 20; }
};

struct Overrider : Base {
  // CHECK-DAG: define linkonce_odr spir_func void @_ZN9Overrider12BaseWithAttrEv{{.*}}#[[ATTRS_INDIR_CALL]]
  [[intel::device_indirectly_callable]] void BaseWithAttr() override { int a = 20; }
  // CHECK-DAG: define linkonce_odr spir_func void @_ZN9Overrider15BaseWithoutAttrEv{{.*}}#[[ATTRS_INDIR_CALL]]
  [[intel::device_indirectly_callable]] void BaseWithoutAttr() override { int b = 30; }
};

struct Overrider1 : Base {
  // CHECK-NOT: define linkonce_odr spir_func void @_ZN10Overrider112BaseWithAttrEv
  void BaseWithAttr() override { int a = 20; }
};

struct Finalizer : Base {
  // CHECK-DAG: define linkonce_odr spir_func void @_ZN9Finalizer12BaseWithAttrEv{{.*}}#[[ATTRS_INDIR_CALL]]
  [[intel::device_indirectly_callable]] void BaseWithAttr() final { int a = 20; }
  // CHECK-DAG: define linkonce_odr spir_func void @_ZN9Finalizer15BaseWithoutAttrEv{{.*}}#[[ATTRS_INDIR_CALL]]
  [[intel::device_indirectly_callable]] void BaseWithoutAttr() final { int b = 30; }
};

struct Finalizer1 : Base {
  // CHECK-NOT: define linkonce_odr spir_func void @_ZN10Finalizer112BaseWithAttrEv
  void BaseWithAttr() final { int a = 20; }
};

// CHECK: attributes #[[ATTRS_INDIR_CALL]] = { {{.*}} "referenced-indirectly"
// CHECK-NOT: attributes #[[ATTRS_NOT_INDIR_CALL]] = { {{.*}} "referenced-indirectly"
// CHECK: attributes #[[ATTRS_INDIR_CALL_1]] = { {{.*}} "referenced-indirectly"
