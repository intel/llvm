// RUN: %clangxx -fsycl -fsyntax-only %s

// Test for a workaround to a bug in clang causing some constexpr lambda
// expressions to not be identified as returning a bool.

#include <sycl/detail/kernel_properties.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

int main() {
  if constexpr (false) {
    sycl::ext::oneapi::experimental::properties prop{
        sycl::ext::intel::experimental::grf_size<256>};
  }
}
