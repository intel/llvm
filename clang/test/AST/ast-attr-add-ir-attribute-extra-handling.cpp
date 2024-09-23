// RUN: %clang_cc1 -fsycl-is-device -std=gnu++11 -ast-dump %s | FileCheck %s

// add_ir_attributes_function attribute used to represent compile-time SYCL
// properties and some of those properties are intended to be turned into
// attributes to enable various diagnostics.
//
// This test is intended to check one (and only, at least for now) of such
// tranformations: property with "indirectly-callable" key should have the same
// effect as applying sycl_device attribute and the test checks that we do add
// that attribute implicitly.

// CHECK-LABEL: ToBeTurnedIntoDeviceFunction 'void ()'
// CHECK: SYCLAddIRAttributesFunctionAttr
// CHECK: SYCLDeviceAttr {{.*}} Implicit
[[__sycl_detail__::add_ir_attributes_function("indirectly-callable", "void")]]
void ToBeTurnedIntoDeviceFunction();

// CHECK-LABEL: NotToBeTurnedIntoDeviceFunction 'void ()'
// CHECK-NOT: SYCLDeviceAttr {{.*}} Implicit
// CHECK:     SYCLAddIRAttributesFunctionAttr
// CHECK-NOT: SYCLDeviceAttr {{.*}} Implicit
[[__sycl_detail__::add_ir_attributes_function("not-indirectly-callable", "void")]]
void NotToBeTurnedIntoDeviceFunction();

template <int V>
struct Metadata {
  static constexpr const char *name = "not-indirectly-callable";
  static constexpr const char *value = "void";
};

template <>
struct Metadata<42> {
  static constexpr const char *name = "indirectly-callable";
  static constexpr const char *value = "void";
};

// CHECK-LABEL: ToBeTurnedIntoDeviceFunctionAttrTemplateArgs 'void ()'
// CHECK: SYCLAddIRAttributesFunctionAttr
// CHECK: SYCLDeviceAttr {{.*}} Implicit
[[__sycl_detail__::add_ir_attributes_function(Metadata<42>::name, Metadata<42>::value)]]
void ToBeTurnedIntoDeviceFunctionAttrTemplateArgs();

// CHECK-LABEL: NotToBeTurnedIntoDeviceFunctionAttrTemplateArgs 'void ()'
// CHECK-NOT: SYCLDeviceAttr {{.*}} Implicit
// CHECK:     SYCLAddIRAttributesFunctionAttr
// CHECK-NOT: SYCLDeviceAttr {{.*}} Implicit
[[__sycl_detail__::add_ir_attributes_function(Metadata<1>::name, Metadata<1>::value)]]
void NotToBeTurnedIntoDeviceFunctionAttrTemplateArgs();

// CHECK-LABEL: class Base definition
class Base {
  // CHECK-LABEL: ToBeTurnedIntoDeviceFunctionAttrTemplateArgs 'void ()'
  // CHECK: SYCLAddIRAttributesFunctionAttr
  // CHECK: SYCLDeviceAttr {{.*}} Implicit
  [[__sycl_detail__::add_ir_attributes_function(Metadata<42>::name, Metadata<42>::value)]]
  virtual void ToBeTurnedIntoDeviceFunctionAttrTemplateArgs();

  // CHECK-LABEL: NotToBeTurnedIntoDeviceFunctionAttrTemplateArgs 'void ()'
  // CHECK-NOT: SYCLDeviceAttr {{.*}} Implicit
  // CHECK:     SYCLAddIRAttributesFunctionAttr
  // CHECK-NOT: SYCLDeviceAttr {{.*}} Implicit
  [[__sycl_detail__::add_ir_attributes_function(Metadata<1>::name, Metadata<1>::value)]]
  virtual void NotToBeTurnedIntoDeviceFunctionAttrTemplateArgs();
};

// CHECK-LABEL: class Derived definition
class Derived : public Base {
  // CHECK-LABEL: ToBeTurnedIntoDeviceFunctionAttrTemplateArgs 'void ()'
  // CHECK: SYCLAddIRAttributesFunctionAttr
  // CHECK: SYCLDeviceAttr {{.*}} Implicit
  [[__sycl_detail__::add_ir_attributes_function(Metadata<42>::name, Metadata<42>::value)]]
  void ToBeTurnedIntoDeviceFunctionAttrTemplateArgs() override;

  // CHECK-LABEL: NotToBeTurnedIntoDeviceFunctionAttrTemplateArgs 'void ()'
  // CHECK-NOT: SYCLDeviceAttr {{.*}} Implicit
  // CHECK:     SYCLAddIRAttributesFunctionAttr
  // CHECK-NOT: SYCLDeviceAttr {{.*}} Implicit
  [[__sycl_detail__::add_ir_attributes_function(Metadata<1>::name, Metadata<1>::value)]]
  void NotToBeTurnedIntoDeviceFunctionAttrTemplateArgs() override;
};

// CHECK-LABEL: class SubDerived definition
class SubDerived : public Derived {
  // CHECK-LABEL: ToBeTurnedIntoDeviceFunctionAttrTemplateArgs 'void ()'
  // CHECK: SYCLAddIRAttributesFunctionAttr
  // CHECK: SYCLDeviceAttr {{.*}} Implicit
  [[__sycl_detail__::add_ir_attributes_function(Metadata<2>::name, Metadata<42>::name, Metadata<2>::value, Metadata<42>::value)]]
  void ToBeTurnedIntoDeviceFunctionAttrTemplateArgs() override;

  // CHECK-LABEL: NotToBeTurnedIntoDeviceFunctionAttrTemplateArgs 'void ()'
  // CHECK-NOT: SYCLDeviceAttr {{.*}} Implicit
  // CHECK:     SYCLAddIRAttributesFunctionAttr
  // CHECK-NOT: SYCLDeviceAttr {{.*}} Implicit
  [[__sycl_detail__::add_ir_attributes_function(Metadata<1>::name, Metadata<2>::name, Metadata<1>::value, Metadata<2>::value)]]
  void NotToBeTurnedIntoDeviceFunctionAttrTemplateArgs() override;
};
