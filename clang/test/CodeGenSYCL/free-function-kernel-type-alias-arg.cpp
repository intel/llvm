// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -sycl-std=2020 -fsycl-int-header=%t.h %s
// RUN: FileCheck -input-file=%t.h %s
//
// The purpose of this test is to ensure that forward declarations of free
// function kernels are emitted properly.
// However, this test checks a specific scenario:
// - free function arguments are type aliases (through using or typedef)

namespace ns {

using IntUsing = int;
typedef int IntTypedef;

template <typename T>
struct Foo {};

template <typename T>
using FooUsing = Foo<T>;

using FooIntUsing = Foo<int>;
typedef Foo<int> FooIntTypedef;

template <typename T1, typename T2>
struct Bar {};

template<typename T1>
using BarUsing = Bar<T1, float>;

template<typename T1, typename T2>
using BarUsing2 = Bar<Foo<T2>, T1>;

template <typename T1>
using BarUsingBarUsing2 = BarUsing2<T1, int>;

template <typename T1>
using BarUsingFooIntUsing = Bar<FooIntUsing, T1>;

template <typename T1>
using BarUsingBarUsingFooIntUsing = BarUsingFooIntUsing<T1>;

class Baz {
public:
  using type = BarUsing<double>;
};

template <template <typename> class T1>
class Foz {};

template <template <typename> class T1>
using FozUsing = Foz<T1>;

using FozFooUsing = Foz<Foo>;

template <typename T1, typename T2, typename T3 = BarUsing2<T1, T2>,
          typename T4 = BarUsing<T2>>
struct AliasAsDefaultArg {};

template <typename T>
struct List {};

template <typename T>
struct Array {};

} // namespace ns

[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void int_using(ns::IntUsing Arg) {}

// CHECK: void int_using(int Arg);

[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void int_typedef(ns::IntTypedef Arg) {}

// CHECK: void int_typedef(int Arg);

[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void foo_using(ns::FooIntUsing Arg) {}

// CHECK: void foo_using(ns::Foo<int> Arg);

[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void foo_typedef(ns::FooIntTypedef Arg) {}

// CHECK: void foo_typedef(ns::Foo<int> Arg);

template<typename T>
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void bar_using(ns::BarUsing<T> Arg) {}
template void bar_using(ns::BarUsing<int>);

// CHECK: template <typename T> void bar_using(ns::Bar<T, float>);

template<typename T1, typename T2>
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void bar_using2(ns::BarUsing2<T1, T2> Arg) {}
template void bar_using2(ns::BarUsing2<int, float>);

// CHECK: template <typename T1, typename T2> void bar_using2(ns::Bar<ns::Foo<T2>, T1>);

template<typename T1>
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void bar_using_bar_using2(ns::BarUsingBarUsing2<T1> Arg) {}
template void bar_using_bar_using2(ns::BarUsingBarUsing2<int>);

// CHECK: template <typename T1> void bar_using_bar_using2(ns::Bar<ns::Foo<int>, T1>);

template<typename T>
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void bar_using_foo_int_using(ns::BarUsingFooIntUsing<T> Arg) {}
template void bar_using_foo_int_using(ns::BarUsingFooIntUsing<float>);

// CHECK: template <typename T> void bar_using_foo_int_using(ns::Bar<ns::Foo<int>, T>);

template<typename T>
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void bar_using_bar_using_foo_int_using(ns::BarUsingBarUsingFooIntUsing<T> Arg) {}
template void bar_using_bar_using_foo_int_using(ns::BarUsingBarUsingFooIntUsing<float>);

// CHECK: template <typename T> void bar_using_bar_using_foo_int_using(ns::Bar<ns::Foo<int>, T>);

[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void baz_type(ns::Baz::type Arg) {}

// CHECK: void baz_type(ns::Bar<double, float> Arg);

[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void foz_using(ns::FozUsing<ns::Foo> Arg) {}

// CHECK: void foz_using(ns::Foz<ns::Foo> Arg);

[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void foz_foo_using(ns::FozFooUsing Arg) {}

// CHECK: void foz_foo_using(ns::Foz<ns::Foo> Arg);

template<template <typename> class T1, typename T2>
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void bar_using2_foz(ns::BarUsing2<ns::Foz<T1>, T2> Arg) {}
template void bar_using2_foz(ns::BarUsing2<ns::Foz<ns::Foo>, float>);

// CHECK: template <template <typename > class T1, typename T2> void bar_using2_foz(ns::Bar<ns::Foo<T2>, ns::Foz<T1>>)

template<typename T1, typename T2>
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void alias_as_default_template_arg(ns::AliasAsDefaultArg<T1, T2> Arg) {}
template void alias_as_default_template_arg(ns::AliasAsDefaultArg<int, float>);

// CHECK: template <typename T1, typename T2> void alias_as_default_template_arg(ns::AliasAsDefaultArg<T1, T2, ns::Bar<ns::Foo<T2>, T1>, ns::Bar<T2, float>>);

template<typename T, template <typename> typename Container>
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void template_template_arg(Container<T> Arg) {}
template void template_template_arg(ns::List<int>);

// CHECK: template <typename T, template <typename > typename Container> void template_template_arg(Container<T>);

// These test cases fail, but they are added here in advance to add a record of
// known bugs.
#if 0

[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void foz_foo_using(ns::Foz<ns::FooUsing> Arg) {}

// CHECK-DISABLED: void foz_foo_using(ns::Foz<ns::Foo> Arg);

[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void foz_using_foo_using(ns::FozUsing<ns::FooUsing> Arg) {}

// CHECK-DISABLED: void foz_using_foo_using(ns::Foz<ns::Foo> Arg);
#endif
