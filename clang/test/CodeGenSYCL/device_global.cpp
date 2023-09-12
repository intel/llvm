// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -disable-llvm-passes -fsycl-unique-prefix=THE_PREFIX -std=c++17 -fgpu-rdc -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-RDC
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -disable-llvm-passes -fsycl-unique-prefix=THE_PREFIX -std=c++17 -fno-gpu-rdc -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-NORDC
#include "sycl.hpp"

// Test cases below show that 'sycl-unique-id' LLVM IR attribute is attached to the
// global variable whose type is decorated with device_global attribute, and that a
// unique string is generated.

using namespace sycl::ext::oneapi;
using namespace sycl;
queue q;

device_global<int> A;
#ifdef SYCL_EXTERNAL
SYCL_EXTERNAL device_global<int> AExt;
#endif
static device_global<int> B;

struct Foo {
  static device_global<int> C;
};
device_global<int> Foo::C;
// CHECK-RDC: @AExt = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global" zeroinitializer, align 8 #[[AEXT_ATTRS:[0-9]+]]
// CHECK: @A = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global" zeroinitializer, align 8 #[[A_ATTRS:[0-9]+]]
// CHECK: @_ZL1B = internal addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global" zeroinitializer, align 8 #[[B_ATTRS:[0-9]+]]
// CHECK: @_ZN3Foo1CE = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global" zeroinitializer, align 8 #[[C_ATTRS:[0-9]+]]

device_global<int> same_name;
namespace NS {
device_global<int> same_name;
}
// CHECK: @same_name = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global" zeroinitializer, align 8 #[[SAME_NAME_ATTRS:[0-9]+]]
// CHECK: @_ZN2NS9same_nameE = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global" zeroinitializer, align 8 #[[SAME_NAME_NS_ATTRS:[0-9]+]]

// decorated with only global_variable_allowed attribute
template <typename T>
class [[__sycl_detail__::global_variable_allowed]] only_global_var_allowed {
public:
  const T &get() const noexcept { return *Data; }
  only_global_var_allowed() {}
  operator T &() noexcept { return *Data; }

private:
  T *Data;
};

// check that we don't generate `sycl-unique-id` IR attribute if class does not use
// [[__sycl_detail__::device_global]]
only_global_var_allowed<int> no_device_global;
// CHECK-RDC: @no_device_global = linkonce_odr addrspace(1) global %class.only_global_var_allowed zeroinitializer, align 8{{$}}
// CHECK-NORDC: @no_device_global = internal addrspace(1) global %class.only_global_var_allowed zeroinitializer, align 8{{$}}

inline namespace Bar {
device_global<float> InlineNS;
}
// CHECK: @_ZN3Bar8InlineNSE = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global.0" zeroinitializer, align 8 #[[BAR_INLINENS_ATTRS:[0-9]+]]

template <typename T> struct TS {
public:
  static device_global<T> d;
};
template <> device_global<int> TS<int>::d{};
// CHECK: @_ZN2TSIiE1dE = addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global" zeroinitializer, align 8 #[[TEMPLATED_WRAPPER_ATTRS:[0-9]+]]

template <typename T>
device_global<T> templ_dev_global;
// CHECK: @[[TEMPL_DEV_GLOB:[a-zA-Z0-9_]+]] = linkonce_odr addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global" zeroinitializer, comdat, align 8 #[[TEMPL_DEV_GLOB_ATTRS:[0-9]+]]

void foo() {
  q.submit([&](handler &h) {
    h.single_task<class kernel_name_1>([=]() {
      (void)A;
      (void)B;
      (void)Foo::C;
      (void)same_name;
      (void)NS::same_name;
      (void)no_device_global;
      (void)Bar::InlineNS;
      auto AA = TS<int>::d.get();
      auto val = templ_dev_global<int>.get();
    });
  });
}

namespace {
device_global<int> same_name;
}
// CHECK: @_ZN12_GLOBAL__N_19same_nameE = internal addrspace(1) global %"class.sycl::_V1::ext::oneapi::device_global" zeroinitializer, align 8 #[[SAME_NAME_ANON_NS_ATTRS:[0-9]+]]

namespace {
void bar() {
  q.submit([&](handler &h) {
    h.single_task<class kernel_name>([=]() { int A = same_name; });
  });
}
} // namespace

// CHECK: @llvm.global_ctors = appending global [2 x { i32, ptr, ptr addrspace(4) }] [{ i32, ptr, ptr addrspace(4) } { i32 65535, ptr @__cxx_global_var_init{{.*}}, ptr addrspace(4) addrspacecast (ptr addrspace(1) @[[TEMPL_DEV_GLOB]] to ptr addrspace(4)) }, { i32, ptr, ptr addrspace(4) } { i32 65535, ptr @_GLOBAL__sub_I_device_global.cpp, ptr addrspace(4) null }]
// CHECK: @llvm.used = appending global [1 x ptr addrspace(4)] [ptr addrspace(4) addrspacecast (ptr addrspace(1) @[[TEMPL_DEV_GLOB]] to ptr addrspace(4))], section "llvm.metadata"
// CHECK: @llvm.compiler.used = appending global [2 x ptr addrspace(4)]
// CHECK-SAME: @_ZL1B
// CHECK-SAME: @_ZN12_GLOBAL__N_19same_nameE

// CHECK-RDC: attributes #[[AEXT_ATTRS]] = { "sycl-unique-id"="_Z4AExt" }
// CHECK: attributes #[[A_ATTRS]] = { "sycl-unique-id"="_Z1A" }
// CHECK: attributes #[[B_ATTRS]] = { "sycl-unique-id"="THE_PREFIX____ZL1B" }
// CHECK: attributes #[[C_ATTRS]] = { "sycl-unique-id"="_ZN3Foo1CE" }
// CHECK: attributes #[[SAME_NAME_ATTRS]] = { "sycl-unique-id"="_Z9same_name" }
// CHECK: attributes #[[SAME_NAME_NS_ATTRS]] = { "sycl-unique-id"="_ZN2NS9same_nameE" }
// CHECK: attributes #[[BAR_INLINENS_ATTRS]] = { "sycl-unique-id"="_ZN3Bar8InlineNSE" }
// CHECK: attributes #[[TEMPLATED_WRAPPER_ATTRS]] = { "sycl-unique-id"="_ZN2TSIiE1dE" }
// CHECK: attributes #[[TEMPL_DEV_GLOB_ATTRS]] = { "sycl-unique-id"="_Z16templ_dev_globalIiE" }
// CHECK: attributes #[[SAME_NAME_ANON_NS_ATTRS]] = { "sycl-unique-id"="THE_PREFIX____ZN12_GLOBAL__N_19same_nameE" }
