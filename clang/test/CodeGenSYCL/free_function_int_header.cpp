// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -sycl-std=2020 -fsycl-int-header=%t.h %s
// RUN: FileCheck -input-file=%t.h %s
// 
// This test checks integration header contents for free functions with scalar
// and pointer parameters.

#include "mock_properties.hpp"
#include "sycl.hpp"

// First overload of function ff_2.
__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel",
                                              2)]] void
ff_2(int *ptr, int start, int end) {
  for (int i = start; i <= end; i++)
    ptr[i] = start + 66;
}

// Second overload of function ff_2.
__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel",
  2)]] void
  ff_2(int* ptr, int start, int end, int value) {
  for (int i = start; i <= end; i++)
    ptr[i] = start + value;
}

// Templated definition of function ff_3.
template <typename T>
__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]] void
ff_3(T *ptr, T start, T end) {
  for (int i = start; i <= end; i++)
    ptr[i] = start;
}

// Explicit instantiation of ff_3 with int type.
template void ff_3(int *ptr, int start, int end);

// Explicit instantiation of ff_3 with float type.
template void ff_3(float* ptr, float start, float end);

// Specialization of ff_3 with double type.
template <> void ff_3<double>(double *ptr, double start, double end) {
  for (int i = start; i <= end; i++)
    ptr[i] = end;
}

// CHECK:      const char* const kernel_names[] = {
// CHECK-NEXT:   {{.*}}__sycl_kernel_ff_2Piii
// CHECK-NEXT:   {{.*}}__sycl_kernel_ff_2Piiii
// CHECK-NEXT:   {{.*}}__sycl_kernel_ff_3IiEvPT_S0_S0_
// CHECK-NEXT:   {{.*}}__sycl_kernel_ff_3IfEvPT_S0_S0_
// CHECK-NEXT:   {{.*}}__sycl_kernel_ff_3IdEvPT_S0_S0_
// CHECK-NEXT: };

// CHECK:      const kernel_param_desc_t kernel_signatures[] = {
// CHECK-NEXT:   {{.*}}__sycl_kernel_ff_2Piii
// CHECK-NEXT:   { kernel_param_kind_t::kind_pointer, 8, 0 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 8 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 12 },

// CHECK:        {{.*}}__sycl_kernel_ff_2Piiii
// CHECK-NEXT:   { kernel_param_kind_t::kind_pointer, 8, 0 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 8 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 12 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 16 },

// CHECK:        {{.*}}__sycl_kernel_ff_3IiEvPT_S0_S0_
// CHECK-NEXT:   { kernel_param_kind_t::kind_pointer, 8, 0 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 8 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 12 },

// CHECK:        {{.*}}__sycl_kernel_ff_3IfEvPT_S0_S0_
// CHECK-NEXT:   { kernel_param_kind_t::kind_pointer, 8, 0 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 8 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 4, 12 },

// CHECK:        {{.*}}__sycl_kernel_ff_3IdEvPT_S0_S0_
// CHECK-NEXT:   { kernel_param_kind_t::kind_pointer, 8, 0 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 8, 8 },
// CHECK-NEXT:   { kernel_param_kind_t::kind_std_layout, 8, 16 },

// CHECK:        { kernel_param_kind_t::kind_invalid, -987654321, -987654321 },
// CHECK-NEXT: };

// CHECK: Definition of _Z18__sycl_kernel_ff_2Piii as a free function kernel
// CHECK-NEXT: void ff_2(int *ptr, int start, int end);
// CHECK-NEXT: static constexpr auto __sycl_shim1() {
// CHECK-NEXT:   return (void (*)(int *, int, int))ff_2;
// CHECK-NEXT: }
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT:   template <>
// CHECK-NEXT:   struct ext::oneapi::experimental::is_kernel<__sycl_shim1()> {
// CHECK-NEXT:     static constexpr bool value = true;
// CHECK-NEXT:   };
// CHECK-NEXT:   template <>
// CHECK-NEXT:   struct ext::oneapi::experimental::is_single_task_kernel<__sycl_shim1()> {
// CHECK-NEXT:     static constexpr bool value = true;
// CHECK-NEXT:   };
// CHECK-NEXT: }

// CHECK: Definition of _Z18__sycl_kernel_ff_2Piiii as a free function kernel
// CHECK-NEXT: void ff_2(int *ptr, int start, int end, int value);
// CHECK-NEXT: static constexpr auto __sycl_shim2() {
// CHECK-NEXT:   return (void (*)(int *, int, int, int))ff_2;
// CHECK-NEXT: }
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT:   template <>
// CHECK-NEXT:   struct ext::oneapi::experimental::is_kernel<__sycl_shim2()> {
// CHECK-NEXT:     static constexpr bool value = true;
// CHECK-NEXT:   };
// CHECK-NEXT:   template <>
// CHECK-NEXT:   struct ext::oneapi::experimental::is_single_task_kernel<__sycl_shim2()> {
// CHECK-NEXT:     static constexpr bool value = true;
// CHECK-NEXT:   };
// CHECK-NEXT: }

// CHECK: Definition of _Z18__sycl_kernel_ff_3IiEvPT_S0_S0_ as a free function kernel
// CHECK-NEXT: template <typename T> void ff_3(T *ptr, T start, T end);
// CHECK-NEXT: static constexpr auto __sycl_shim3() {
// CHECK-NEXT:   return (void (*)(int *, int, int))ff_3<int>;
// CHECK-NEXT: }
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT:   template <>
// CHECK-NEXT:   struct ext::oneapi::experimental::is_kernel<__sycl_shim3()> {
// CHECK-NEXT:     static constexpr bool value = true;
// CHECK-NEXT:   };
// CHECK-NEXT:   template <>
// CHECK-NEXT:   struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim3(), 2> {
// CHECK-NEXT:     static constexpr bool value = true;
// CHECK-NEXT:   };
// CHECK-NEXT: }
 
// CHECK: Definition of _Z18__sycl_kernel_ff_3IfEvPT_S0_S0_ as a free function kernel
// CHECK-NEXT: template <typename T> void ff_3(T *ptr, T start, T end);
// CHECK-NEXT: static constexpr auto __sycl_shim4() {
// CHECK-NEXT:   return (void (*)(float *, float, float))ff_3<float>;
// CHECK-NEXT: }
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT:   template <>
// CHECK-NEXT:   struct ext::oneapi::experimental::is_kernel<__sycl_shim4()> {
// CHECK-NEXT:     static constexpr bool value = true;
// CHECK-NEXT:   };
// CHECK-NEXT:   template <>
// CHECK-NEXT:   struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim4(), 2> {
// CHECK-NEXT:     static constexpr bool value = true;
// CHECK-NEXT:   };
// CHECK-NEXT: }

// CHECK: Definition of _Z18__sycl_kernel_ff_3IdEvPT_S0_S0_ as a free function kernel
// CHECK-NEXT: template <typename T> void ff_3(T *ptr, T start, T end);
// CHECK-NEXT: static constexpr auto __sycl_shim5() {
// CHECK-NEXT:   return (void (*)(double *, double, double))ff_3<double>;
// CHECK-NEXT: }
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT:   template <>
// CHECK-NEXT:   struct ext::oneapi::experimental::is_kernel<__sycl_shim5()> {
// CHECK-NEXT:     static constexpr bool value = true;
// CHECK-NEXT:   };
// CHECK-NEXT:   template <>
// CHECK-NEXT:   struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim5(), 2> {
// CHECK-NEXT:     static constexpr bool value = true;
// CHECK-NEXT:   };
// CHECK-NEXT: }

// CHECK: #include <sycl/kernel_bundle.hpp>

// CHECK: Definition of kernel_id of _Z18__sycl_kernel_ff_2Piii
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT:   template <>
// CHECK-NEXT:   kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim1()>() {
// CHECK-NEXT:     return sycl::detail::get_kernel_id_impl(std::string_view{"_Z18__sycl_kernel_ff_2Piii"});
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK: Definition of kernel_id of _Z18__sycl_kernel_ff_2Piiii
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT:   template <>
// CHECK-NEXT:   kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim2()>() {
// CHECK-NEXT:     return sycl::detail::get_kernel_id_impl(std::string_view{"_Z18__sycl_kernel_ff_2Piiii"});
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK: Definition of kernel_id of _Z18__sycl_kernel_ff_3IiEvPT_S0_S0_
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT:   template <>
// CHECK-NEXT:   kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim3()>() {
// CHECK-NEXT:     return sycl::detail::get_kernel_id_impl(std::string_view{"_Z18__sycl_kernel_ff_3IiEvPT_S0_S0_"});
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK: Definition of kernel_id of _Z18__sycl_kernel_ff_3IfEvPT_S0_S0_
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT:   template <>
// CHECK-NEXT:   kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim4()>() {
// CHECK-NEXT:     return sycl::detail::get_kernel_id_impl(std::string_view{"_Z18__sycl_kernel_ff_3IfEvPT_S0_S0_"});
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK: Definition of kernel_id of _Z18__sycl_kernel_ff_3IdEvPT_S0_S0_
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT:   template <>
// CHECK-NEXT:   kernel_id ext::oneapi::experimental::get_kernel_id<__sycl_shim5()>() {
// CHECK-NEXT:     return sycl::detail::get_kernel_id_impl(std::string_view{"_Z18__sycl_kernel_ff_3IdEvPT_S0_S0_"});
// CHECK-NEXT:   }
// CHECK-NEXT: }