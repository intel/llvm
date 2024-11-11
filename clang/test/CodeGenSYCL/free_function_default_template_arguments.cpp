// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -sycl-std=2020 -fsycl-int-header=%t.h %s
// RUN: FileCheck -input-file=%t.h %s

// This test checks integration header contents free functions kernels in
// presense of template arguments.

#include "mock_properties.hpp"
#include "sycl.hpp"

namespace ns {

struct notatuple {
  int a;
};

template <typename T, typename = int, int a = 12, typename = notatuple, typename ...TS> struct Arg {
  T val;
};

[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel",
                                              2)]] void
simple(Arg<char>){
}

}


template <typename T>
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]] void
templated(ns::Arg<T, float, 3>, T end) {
}

template void templated(ns::Arg<int, float, 3>, int);

using namespace ns;

template <typename T>
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]] void
templated2(Arg<T, notatuple>, T end) {
}

template void templated2(Arg<int, notatuple>, int);

// CHECK: Definition of _ZN16__sycl_kernel_ns6simpleENS_3ArgIciLi12ENS_9notatupleEJEEE as a free function kernel

// CHECK: Forward declarations of kernel and its argument types:
// CHECK-NEXT: namespace ns { 
// CHECK-NEXT: struct notatuple;
// CHECK-NEXT: }
// CHECK-NEXT: namespace ns { 
// CHECK-NEXT: template <typename T, typename, int a, typename, typename ...TS> struct Arg;
// CHECK-NEXT: }

// CHECK: void ns::simple(ns::Arg<char, int, 12, ns::notatuple>);
// CHECK-NEXT: static constexpr auto __sycl_shim1() {
// CHECK-NEXT:   return (void (*)(struct ns::Arg<char, int, 12, struct ns::notatuple>))simple;
// CHECK-NEXT: }

// CHECK: Definition of _Z23__sycl_kernel_templatedIiEvN2ns3ArgIT_fLi3ENS0_9notatupleEJEEES2_ as a free function kernel
// CHECK: template <typename T> void templated(ns::Arg<T, float, 3, ns::notatuple>, T end);
// CHECK-NEXT: static constexpr auto __sycl_shim2() {
// CHECK-NEXT:   return (void (*)(struct ns::Arg<int, float, 3, struct ns::notatuple>, int))templated<int>;
// CHECK-NEXT: }

// CHECK: Definition of _Z24__sycl_kernel_templated2IiEvN2ns3ArgIT_NS0_9notatupleELi12ES3_JEEES2_ as a free function kernel
// CHECK: Forward declarations of kernel and its argument types:
// CHECK: template <typename T> void templated2(ns::Arg<T, ns::notatuple, 12, ns::notatuple>, T end);
// CHECK-NEXT: static constexpr auto __sycl_shim3() {
// CHECK-NEXT:   return (void (*)(struct ns::Arg<int, struct ns::notatuple, 12, struct ns::notatuple>, int))templated2<int>;
// CHECK-NEXT: }
