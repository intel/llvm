// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -fsycl-unique-prefix=THE_PREFIX -std=c++17 -emit-llvm %s -o - | FileCheck %s
#include "sycl.hpp"

// Test cases below show that 'sycl-unique-id' LLVM IR attribute is attached to the
// global variable whose type is decorated with device_global attribute, and that a
// unique string is generated.

using namespace sycl::ext::oneapi;
using namespace cl::sycl;
queue q;

device_global<int> A;
static device_global<int> B;

struct Foo {
  static device_global<int> C;
};
device_global<int> Foo::C;
// CHECK: @A = addrspace(1) global %"class.cl::sycl::ext::oneapi::device_global" zeroinitializer, align 8 #[[A_ATTRS:[0-9]+]]
// CHECK: @_ZL1B = internal addrspace(1) global %"class.cl::sycl::ext::oneapi::device_global" zeroinitializer, align 8 #[[B_ATTRS:[0-9]+]]
// CHECK: @_ZN3Foo1CE = addrspace(1) global %"class.cl::sycl::ext::oneapi::device_global" zeroinitializer, align 8 #[[C_ATTRS:[0-9]+]]

device_global<int> same_name;
namespace NS {
device_global<int> same_name;
}
// CHECK: @same_name = addrspace(1) global %"class.cl::sycl::ext::oneapi::device_global" zeroinitializer, align 8 #[[SAME_NAME_ATTRS:[0-9]+]]
// CHECK: @_ZN2NS9same_nameE = addrspace(1) global %"class.cl::sycl::ext::oneapi::device_global" zeroinitializer, align 8 #[[SAME_NAME_NS_ATTRS:[0-9]+]]

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
// CHECK: @no_device_global = addrspace(1) global %class.only_global_var_allowed zeroinitializer, align 8{{$}}

inline namespace Bar {
device_global<float> InlineNS;
}
// CHECK: @_ZN3Bar8InlineNSE = addrspace(1) global %"class.cl::sycl::ext::oneapi::device_global.0" zeroinitializer, align 8 #[[BAR_INLINENS_ATTRS:[0-9]+]]

template <typename T> struct TS {
public:
  static device_global<T> d;
};
template <> device_global<int> TS<int>::d{};
// CHECK: @_ZN2TSIiE1dE = addrspace(1) global %"class.cl::sycl::ext::oneapi::device_global" zeroinitializer, align 8 #[[TEMPLATED_WRAPPER_ATTRS:[0-9]+]]

template <typename T>
device_global<T> templ_dev_global;
// CHECK: @[[TEMPL_DEV_GLOB:[a-zA-Z0-9_]+]] = linkonce_odr addrspace(1) global %"class.cl::sycl::ext::oneapi::device_global" zeroinitializer, comdat, align 8 #[[TEMPL_DEV_GLOB_ATTRS:[0-9]+]]

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
// CHECK: @_ZN12_GLOBAL__N_19same_nameE = internal addrspace(1) global %"class.cl::sycl::ext::oneapi::device_global" zeroinitializer, align 8 #[[SAME_NAME_ANON_NS_ATTRS:[0-9]+]]

namespace {
void bar() {
  q.submit([&](handler &h) {
    h.single_task<class kernel_name>([=]() { int A = same_name; });
  });
}
} // namespace

// CHECK: @llvm.global_ctors = appending global [2 x { i32, void ()*, i8 addrspace(4)* }] [{ i32, void ()*, i8 addrspace(4)* } { i32 65535, void ()* @__cxx_global_var_init{{.*}}, i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast (%"class.cl::sycl::ext::oneapi::device_global" addrspace(1)* @[[TEMPL_DEV_GLOB]] to i8 addrspace(1)*) to i8 addrspace(4)*) }, { i32, void ()*, i8 addrspace(4)* } { i32 65535, void ()* @_GLOBAL__sub_I_device_global.cpp, i8 addrspace(4)* null }]
// CHECK: @llvm.used = appending global [1 x i8 addrspace(4)*] [i8 addrspace(4)* addrspacecast (i8 addrspace(1)* bitcast (%"class.cl::sycl::ext::oneapi::device_global" addrspace(1)* @[[TEMPL_DEV_GLOB]] to i8 addrspace(1)*) to i8 addrspace(4)*)], section "llvm.metadata"

// CHECK: attributes #[[A_ATTRS]] = { "sycl-unique-id"="_Z1A" }
// CHECK: attributes #[[B_ATTRS]] = { "sycl-unique-id"="THE_PREFIX____ZL1B" }
// CHECK: attributes #[[C_ATTRS]] = { "sycl-unique-id"="_ZN3Foo1CE" }
// CHECK: attributes #[[SAME_NAME_ATTRS]] = { "sycl-unique-id"="_Z9same_name" }
// CHECK: attributes #[[SAME_NAME_NS_ATTRS]] = { "sycl-unique-id"="_ZN2NS9same_nameE" }
// CHECK: attributes #[[BAR_INLINENS_ATTRS]] = { "sycl-unique-id"="_ZN3Bar8InlineNSE" }
// CHECK: attributes #[[TEMPLATED_WRAPPER_ATTRS]] = { "sycl-unique-id"="_ZN2TSIiE1dE" }
// CHECK: attributes #[[TEMPL_DEV_GLOB_ATTRS]] = { "sycl-unique-id"="_Z16templ_dev_globalIiE" }
// CHECK: attributes #[[SAME_NAME_ANON_NS_ATTRS]] = { "sycl-unique-id"="THE_PREFIX____ZN12_GLOBAL__N_19same_nameE" }
