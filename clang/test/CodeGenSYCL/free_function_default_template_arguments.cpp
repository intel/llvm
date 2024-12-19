// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -sycl-std=2020 -fsycl-int-header=%t.h %s
// RUN: FileCheck -input-file=%t.h %s

// This test checks integration header contents for free functions kernels with
// parameter types that have default template arguments.

#include "mock_properties.hpp"
#include "sycl.hpp"

namespace ns {

struct notatuple {
  int a;
};

namespace ns1 {
template <typename A = notatuple>
class hasDefaultArg {

};
}

template <typename T, typename = int, int a = 12, typename = notatuple, typename ...TS> struct Arg {
  T val;
};

[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel",
                                              2)]] void
simple(Arg<char>){
}

}

[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel",
                                              2)]] void
simple1(ns::Arg<ns::ns1::hasDefaultArg<>>){
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

template <typename T, int a = 3>
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]] void
templated3(Arg<T, notatuple, a, ns1::hasDefaultArg<>, int, int>, T end) {
}

template void templated3(Arg<int, notatuple, 3, ns1::hasDefaultArg<>, int, int>, int);


namespace sycl {
template <typename T> struct X {};
template <> struct X<int> {};
namespace detail {
struct Y {};
} // namespace detail
template <> struct X<detail::Y> {};
} // namespace sycl
using namespace sycl;
template <typename T, typename = X<detail::Y>> struct Arg1 { T val; };

[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel",
                                              2)]] void
foo(Arg1<int> arg) {
  arg.val = 42;
}

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

// CHECK: Forward declarations of kernel and its argument types:
// CHECK: namespace ns {
// CHECK: namespace ns1 {
// CHECK-NEXT: template <typename A> class hasDefaultArg;
// CHECK-NEXT: }

// CHECK: void simple1(ns::Arg<ns::ns1::hasDefaultArg<ns::notatuple>, int, 12, ns::notatuple>);
// CHECK-NEXT: static constexpr auto __sycl_shim2() {
// CHECK-NEXT:   return (void (*)(struct ns::Arg<class ns::ns1::hasDefaultArg<struct ns::notatuple>, int, 12, struct ns::notatuple>))simple1;
// CHECK-NEXT: }

// CHECK: template <typename T> void templated(ns::Arg<T, float, 3, ns::notatuple>, T end);
// CHECK-NEXT: static constexpr auto __sycl_shim3() {
// CHECK-NEXT:   return (void (*)(struct ns::Arg<int, float, 3, struct ns::notatuple>, int))templated<int>;
// CHECK-NEXT: }

// CHECK: template <typename T> void templated2(ns::Arg<T, ns::notatuple, 12, ns::notatuple>, T end);
// CHECK-NEXT: static constexpr auto __sycl_shim4() {
// CHECK-NEXT:   return (void (*)(struct ns::Arg<int, struct ns::notatuple, 12, struct ns::notatuple>, int))templated2<int>;
// CHECK-NEXT: }

// CHECK: template <typename T, int a> void templated3(ns::Arg<T, ns::notatuple, a, ns::ns1::hasDefaultArg<ns::notatuple>, int, int>, T end);
// CHECK-NEXT: static constexpr auto __sycl_shim5() {
// CHECK-NEXT:   return (void (*)(struct ns::Arg<int, struct ns::notatuple, 3, class ns::ns1::hasDefaultArg<struct ns::notatuple>, int, int>, int))templated3<int, 3>;
// CHECK-NEXT: }

// CHECK Forward declarations of kernel and its argument types:
// CHECK: namespace sycl { namespace detail {
// CHECK-NEXT: struct Y;
// CHECK-NEXT: }}
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: template <typename T> struct X;
// CHECK-NEXT: }
// CHECK-NEXT: template <typename T, typename> struct Arg1;

// CHECK: void foo(Arg1<int, sycl::X<sycl::detail::Y> > arg);
// CHECK-NEXT: static constexpr auto __sycl_shim6() {
// CHECK-NEXT:   return (void (*)(struct Arg1<int, struct sycl::X<struct sycl::detail::Y> >))foo;
// CHECK-NEXT: }
