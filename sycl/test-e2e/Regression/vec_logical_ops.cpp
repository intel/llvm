// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if preview-breaking-changes-supported %{ %clangxx -fsycl -fpreview-breaking-changes && ./%t.out %}

#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;
  q.single_task([=]() {
    auto testVec1 = sycl::vec<double, 16>(static_cast<double>(1));
    auto testVec2 = sycl::vec<double, 16>(static_cast<double>(2));
    sycl::vec<std::int64_t, 16> resVec1 = testVec1 || testVec2;
    sycl::vec<std::int64_t, 16> resVec2 = testVec1 && testVec2;
  });
}
