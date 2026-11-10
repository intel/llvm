// RUN: %clangxx -fsycl -fsyntax-only %s

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;
  Q.single_task([]() {
    sycl::vec<int, 4> X{1};
    static_assert(std::is_same_v<decltype(X.swizzle<0>())::element_type, int>);
    static_assert(std::is_same_v<decltype(X.swizzle<0>())::value_type, int>);
  });
  return 0;
}
