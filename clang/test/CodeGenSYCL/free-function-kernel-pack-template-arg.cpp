// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -sycl-std=2020 -fsycl-int-header=%t.h %s
// RUN: FileCheck -input-file=%t.h %s
//
// The purpose of this test is to ensure that forward declarations of free
// function kernels are emitted properly.
// However, this test checks a specific scenario:
// - parameter packs are emitted correctly.

template<typename... T>
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void parameter_pack(T... Args) {}
template void parameter_pack(int Arg1, float Arg2);

// CHECK: template <typename ... T> void parameter_pack(T...);
