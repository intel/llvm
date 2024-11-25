// RUN: %clangxx -fsycl -fsycl-unnamed-lambda -fsyntax-only %s
#include <sycl/sycl.hpp>

// This validates the case where using a lambda in a kernel in a different order
// than the lexical order of the lambdas. In a previous implementation of
// __builtin_sycl_unique_stable_name this would result in the value of the
// builtin being invalidated, causing a compile error. The redesigned
// implementation should no longer have a problem with this pattern.
int main() {
  auto w = [](auto i) {};
  sycl::queue q;
  q.parallel_for(10, [](auto i) {});
  q.parallel_for(10, w);
}
