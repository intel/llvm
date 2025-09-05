// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -sycl-std=2020 -fsycl-int-header=%t.h %s
// RUN: FileCheck -input-file=%t.h %s
//
// The purpose of this test is to ensure that forward declarations of free
// function kernels are emitted properly.
// However, this test checks a specific scenario:
// - free function kernel is a function template
// - its argument is templated and has non-type template parameter (with default
//   value) that is an enumeration defined within a namespace

namespace ns {

enum class enum_A { A, B, C };

template<typename T, enum_A V = enum_A::B>
class feature_A {};

namespace nested {
enum class enum_B { A, B, C };

template<typename T, int V, enum_B V2 = enum_B::A, enum_A V3 = enum_A::C>
struct feature_B {};
}

inline namespace nested_inline {
namespace nested2 {
enum class enum_C { A, B, C };

template<int V = 42, enum_C V2 = enum_C::B>
struct feature_C {};
}
} // namespace nested_inline
} // namespace ns

template<typename T>
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void templated_on_A(ns::feature_A<T> Arg) {}
template void templated_on_A(ns::feature_A<int>);

// CHECK: template <typename T> void templated_on_A(ns::feature_A<T, ns::enum_A::B>);

template<typename T, int V = 42>
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void templated_on_B(ns::nested::feature_B<T, V> Arg) {}
template void templated_on_B(ns::nested::feature_B<int, 12>);

// CHECK: template <typename T, int V> void templated_on_B(ns::nested::feature_B<T, V, ns::nested::enum_B::A, ns::enum_A::C>);

template<int V>
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void templated_on_C(ns::nested2::feature_C<V> Arg) {}
template void templated_on_C(ns::nested2::feature_C<42>);

// CHECK: template <int V> void templated_on_C(ns::nested2::feature_C<V, ns::nested2::enum_C::B>);
