// RUN: %{build} -DINCLUDE_BEFORE -o %t.out
// RUN: %{run} %t.out
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// RUN: %if linux %{ %{build} -DINCLUDE_BEFORE -fsycl-host-compiler=g++ -o %t.out %}
// RUN: %if linux %{ %{run} %t.out %}
// RUN: %if linux %{ %{build} -fsycl-host-compiler=g++ -o %t.out %}
// RUN: %if linux %{ %{run} %t.out %}

// RUN: %if windows %{ %{build} -DINCLUDE_BEFORE -fsycl-host-compiler=cl -fsycl-host-compiler-options="/std:c++17" -o %t.out %}
// RUN: %if windows %{ %{run} %t.out %}
// RUN: %if windows %{ %{build} -fsycl-host-compiler=cl -fsycl-host-compiler-options="/std:c++17" -o %t.out %}
// RUN: %if windows %{ %{run} %t.out %}

// Test scenario when <complex> is included before SYCL headers.
#ifdef INCLUDE_BEFORE
#include <complex>
#endif

#include <sycl/detail/core.hpp>

#ifndef INCLUDE_BEFORE
#include <complex>
#endif

#include <iostream>
using namespace sycl;

int main() {
  queue q;
  auto test = []() {
    static_assert(sycl::detail::is_complex<std::complex<float>>::value);
    static_assert(sycl::detail::is_complex<const std::complex<half>>::value);
    static_assert(
        sycl::detail::is_complex<const volatile std::complex<float>>::value);
    static_assert(
        sycl::detail::is_complex<volatile std::complex<double>>::value);
    static_assert(sycl::detail::isComplex<std::complex<float>>::value);
#ifdef __SYCL_DEVICE_ONLY__
    static_assert(
        std::is_same_v<sycl::detail::GroupOpTag<std::complex<float>>::type,
                       sycl::detail::GroupOpC>);
#endif
    static_assert(
        std::is_same_v<
            sycl::detail::select_cl_scalar_complex_or_T_t<std::complex<float>>,
            __spv::complex_float>);
    static_assert(
        std::is_same_v<sycl::detail::select_cl_scalar_complex_or_T_t<float>,
                       float>);
  };
  test();
  q.single_task<class KernelName>([=] {
     test();
     std::ignore = std::complex<float>{0.0f, 1.0f};
   }).wait();

  std::cout << "Passed" << std::endl;
  return 0;
}
