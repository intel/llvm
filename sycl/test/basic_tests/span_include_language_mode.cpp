// RUN: %clangxx -fsycl -fsyntax-only -std=c++17 -I %S/Inputs/no_span -Xclang -verify %s
// RUN: %clangxx -fsycl -fsyntax-only -std=c++20 -DCXX20 -Xclang -verify %s
// expected-no-diagnostics

// In C++17, the poison <span> header catches includes even when the standard
// library leaves __cpp_lib_span undefined. C++20 must retain std::span support.

#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/experimental/ipc_event.hpp>
#include <sycl/ext/oneapi/experimental/ipc_memory.hpp>
#include <sycl/ext/oneapi/experimental/ipc_physical_memory.hpp>
#include <sycl/ext/oneapi/experimental/syclbin_kernel_bundle.hpp>
#include <sycl/sycl.hpp>

#ifdef CXX20
#if __has_include(<version>)
#include <version>
#endif
#ifdef __cpp_lib_span
static_assert(std::span<int>::extent == std::dynamic_extent);
#endif
#endif
