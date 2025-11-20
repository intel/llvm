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

using FooIntUsing = Foo<int>;
typedef Foo<int> FooIntTypedef;

template <typename T1, typename T2>
struct Bar {};

template<typename T1>
using BarUsing = Bar<T1, float>;

template<typename T1, typename T2>
using BarUsing2 = Bar<Foo<T2>, T1>;

class Baz {
public:
  using type = BarUsing<double>;
};

template <typename T1, typename T2, typename T3 = BarUsing2<T1, T2>,
          typename T4 = BarUsing<T2>>
struct AliasAsDefaultArg {};

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

[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void baz_type(ns::Baz::type Arg) {}

// CHECK: void baz_type(ns::Bar<double, float> Arg);

#if 0
// This test case fails, but it is added here in advance to add a record of a
// known bug.
template<typename T1, typename T2>
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void alias_as_default_template_arg(ns::AliasAsDefaultArg<T1, T2> Arg) {}
template void alias_as_default_template_arg(ns::AliasAsDefaultArg<int, float>);

// CHECK-DISABLED: template <typename T1, typename T2> void alias_as_default_template_arg(ns::AliasAsDefaultArg<T1, T2, ns::Bar<ns::Foo<T2>, T1>, ns::Bar<T2, float>>);
#endif
