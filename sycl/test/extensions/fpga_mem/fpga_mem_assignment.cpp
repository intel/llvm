// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
// expected-no-diagnostics

// Test assignment works

#include "sycl/sycl.hpp"

namespace intel = sycl::ext::intel::experimental;   // for fpga_mem
namespace oneapi = sycl::ext::oneapi::experimental; // for properties

// Scalar Assignment
const int scalar = 5;
constexpr intel::fpga_mem<int> scalar_int = scalar;
static_assert(scalar_int.get() == 5);

// Copy constructor
static constexpr intel::fpga_mem<int[3], decltype(oneapi::properties(
                                             intel::num_banks<888>))>
    mem1{1, 8, 7};
static constexpr auto mem2 = mem1;
static_assert(mem1[1] == mem2[1]);
static_assert(mem1.has_property<intel::num_banks_key>());
static_assert(mem1.get_property<intel::num_banks_key>().value == 888);
static_assert(mem2.has_property<intel::num_banks_key>());
static_assert(mem2.get_property<intel::num_banks_key>().value == 888);

int main() {
  sycl::queue Q;
  int f = 0;

  Q.single_task(
      [=]() { volatile int ReadVal = scalar_int.get() + mem1[f] + mem2[f]; });
  return 0;
}
