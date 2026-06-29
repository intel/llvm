// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -sycl-std=2020 -fsycl-int-header=%t.h %s
// RUN: FileCheck -input-file=%t.h %s

// Test purpose - if a free function kernel has a template argument - it should
// be forward declared to avoid compilation error.

template <typename scalar_t, typename F>
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 1)]] void
templated(scalar_t *) {
}

template <typename T> struct TestStruct {
  T val;
};

template void templated<float, TestStruct<float>>(float *);

// CHECK: template <typename T> struct TestStruct;
// CHECK-NEXT: template <typename scalar_t, typename F> void templated(scalar_t *);
// CHECK-NEXT: static constexpr auto __sycl_shim1() {
// CHECK-NEXT:   return (void (*)(float *))templated<float, struct TestStruct<float>>;
// CHECK-NEXT: }
