// RUN: %clang_cc1 -fsycl-is-device -std=c++17 -sycl-std=2020 -ast-dump %s | FileCheck %s
#include "Inputs/sycl.hpp"

// Test cases below check that DeviceGlobalAttr and GlobalVariableAllowedAttr
// are correctly emitted.

using namespace sycl::ext::oneapi;

device_global<int> glob;
// CHECK: ClassTemplateDecl {{.*}} device_global
// CHECK: CXXRecordDecl {{.*}} struct device_global definition
// CHECK: SYCLDeviceGlobalAttr {{.*}}
// CHECK: SYCLGlobalVariableAllowedAttr {{.*}}
// CHECK: ClassTemplateSpecializationDecl {{.*}} struct device_global definition
// CHECK: SYCLDeviceGlobalAttr {{.*}}
// CHECK: SYCLGlobalVariableAllowedAttr {{.*}}
