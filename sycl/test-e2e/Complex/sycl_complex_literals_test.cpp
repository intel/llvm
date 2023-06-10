// DEFINE: %{mathflags} = %if cl_options %{/clang:-fno-fast-math%} %else %{-fno-fast-math%}

// RUN: %{build} -fsycl-device-code-split=per_kernel %{mathflags} -o %t.out
// RUN: %{run} %t.out

#include "sycl_complex_helper.hpp"

int main() {
  using namespace sycl::ext::oneapi::experimental::complex_literals;

  bool test_passes = true;

  test_passes &=
      (std::is_same_v<decltype(3.0i), experimental::complex<double>>);
  test_passes &= (std::is_same_v<decltype(3i), experimental::complex<double>>);
  test_passes &=
      (std::is_same_v<decltype(3.0if), experimental::complex<float>>);
  test_passes &= (std::is_same_v<decltype(3if), experimental::complex<float>>);

  if (!test_passes)
    std::cerr << "cmplx literals test failed\n";

  return test_passes;
}
