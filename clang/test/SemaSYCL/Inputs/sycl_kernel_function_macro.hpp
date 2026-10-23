// Mirrors SYCL_EXT_ONEAPI_KERNEL_FUNCTION from
// sycl/ext/oneapi/experimental/free_function_traits.hpp, plus a minimal launch
// function. nd_launch returns its kernel_function selector so tests can check
// which kernel the macro resolved to with decltype.
#pragma once

#define ND_KERNEL                                                              \
  [[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 1)]]   \
  __attribute__((sycl_device))

template <auto *Func> struct kernel_function_s {};
template <auto *Func> constexpr kernel_function_s<Func> kernel_function{};

template <auto *Func, typename... Args>
kernel_function_s<Func> nd_launch(kernel_function_s<Func> K, Args &&...) {
  return K;
}

#define SYCL_EXT_ONEAPI_KERNEL_FUNCTION(NAME, ...)                             \
  kernel_function<__builtin_sycl_launch_kernel(NAME(__VA_ARGS__))>,            \
      ##__VA_ARGS__

template <typename T, typename U> constexpr bool is_same_v = false;
template <typename T> constexpr bool is_same_v<T, T> = true;

// True if the launch expression L resolved to the kernel named by the rest.
#define LAUNCHES(L, ...) is_same_v<decltype(L), kernel_function_s<__VA_ARGS__>>
