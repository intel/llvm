// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// Tests that the right value returned after setting a specialization constant
// of sycl::vec<char, 16> type is correct.

#include <sycl/detail/core.hpp>

#include <sycl/specialization_id.hpp>

#include <algorithm>

constexpr sycl::specialization_id<sycl::vec<char, 16>> spec_const(20);

int main() {
  sycl::vec<char, 16> Result{0};
  sycl::vec<char, 16> Ref{5};
  sycl::queue Q;
  Q.submit([&](sycl::handler &CGH) {
    CGH.set_specialization_constant<spec_const>(Ref);
    Result = CGH.get_specialization_constant<spec_const>();
  });
  auto CompRes = Ref == Result;
  assert(std::all_of(&CompRes[0], &CompRes[0] + 16,
                     [](const bool &A) { return A; }));
  return 0;
}
