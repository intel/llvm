// RUN: %clangxx -fsycl -fsyntax-only %s
// RUN: %clangxx -fsycl -fpreview-breaking-changes -fsyntax-only %s

#include <sycl/sycl.hpp>

int main() {
  auto testVec1 = sycl::vec<double, 16>(static_cast<double>(1));
  auto testVec2 = sycl::vec<double, 16>(static_cast<double>(2));

  sycl::vec<std::int64_t, 16> resVec1 = testVec1 || testVec2;
  sycl::vec<std::int64_t, 16> resVec2 = testVec1 && testVec2;
}
