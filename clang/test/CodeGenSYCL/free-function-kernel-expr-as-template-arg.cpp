// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -sycl-std=2020 -fsycl-int-header=%t.h %s
// RUN: FileCheck -input-file=%t.h %s
//
// The purpose of this test is to ensure that forward declarations of free
// function kernels are emitted properly.
// However, this test checks a specific scenario:
// - free function argument is a template which accepts constant expressions as
//   arguments

constexpr int A = 2;
constexpr int B = 3;

namespace ns {

constexpr int C = 4;

struct Foo {
  static constexpr int D = 5;
};

enum non_class_enum {
  VAL_A,
  VAL_B
};

enum class class_enum {
  VAL_A,
  VAL_B
};

enum non_class_enum_typed : int {
  VAL_C,
  VAL_D
};

enum class class_enum_typed : int {
  VAL_C,
  VAL_D
};

constexpr int bar(int arg) {
  return arg + 42;
}

} // namespace ns

template<int V>
struct Arg {};

template<ns::non_class_enum V>
struct Arg2 {};

template<ns::non_class_enum_typed V>
struct Arg3 {};

template<ns::class_enum V>
struct Arg4 {};

template<ns::class_enum_typed V>
struct Arg5 {};

[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void constant(Arg<1>) {}

// CHECK: void constant(Arg<1> );

[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void constexpr_v(Arg<A>) {}

// CHECK: void constexpr_v(Arg<2> );

[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void constexpr_expr(Arg<A * B>) {}

// CHECK: void constexpr_expr(Arg<6> );

[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void constexpr_ns(Arg<ns::C>) {}

// CHECK: void constexpr_ns(Arg<4> );

[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void constexpr_ns2(Arg<ns::Foo::D>) {}

// CHECK: void constexpr_ns2(Arg<5> );

[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void constexpr_ns2(Arg2<ns::non_class_enum::VAL_A>) {}

// CHECK: void constexpr_ns2(Arg2<ns::VAL_A> );

[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void constexpr_ns2(Arg3<ns::non_class_enum_typed::VAL_C>) {}

// CHECK: void constexpr_ns2(Arg3<ns::VAL_C> );

[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void constexpr_ns2(Arg4<ns::class_enum::VAL_A>) {}

// CHECK: void constexpr_ns2(Arg4<ns::class_enum::VAL_A> );

[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void constexpr_ns2(Arg5<ns::class_enum_typed::VAL_C>) {}

// CHECK: void constexpr_ns2(Arg5<ns::class_enum_typed::VAL_C> );

[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void constexpr_call(Arg<ns::bar(B)>) {}

// CHECK: void constexpr_call(Arg<45> );
