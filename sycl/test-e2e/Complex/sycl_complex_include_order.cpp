// rUN: %{build} -DINCLUDE_BEFORE -o %t.out
// rUN: %{run} %t.out
// rUN: %{build} -o %t.out
// rUN: %{run} %t.out

// RUN: %if linux %{ %{build} -DINCLUDE_BEFORE -fsycl-host-compiler=g++ -o %t.out %}
// RUN: %if linux %{ %{run} %t.out %}
// RUN: %if linux %{ %{build} -fsycl-host-compiler=g++ -o %t.out %}
// RUN: %if linux %{ %{run} %t.out %}

// Test scenario when <complex> is included before SYCL headers.
#ifdef INCLUDE_BEFORE
#include <complex>
#endif

#include <sycl/sycl.hpp>

#ifndef INCLUDE_BEFORE
#include <complex>
#endif

using namespace sycl;

int main() {
  queue q;
  auto test = []() {
    static_assert(sycl::detail::is_complex<std::complex<float>>::value);
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

  return 0;
}
