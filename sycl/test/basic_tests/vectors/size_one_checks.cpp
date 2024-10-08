// RUN: %clangxx -fsycl %s -o %t_default.out
// RUN: %t_default.out
// RUN: %if preview-breaking-changes-supported %{ %clangxx -fsycl -fpreview-breaking-changes %s -o %t_vec.out %}
// RUN: %if preview-breaking-changes-supported %{ %t_vec.out %}

#include <sycl/types.hpp>

int main() {
  sycl::vec<int, 1> v1{42};
  sycl::vec<int, 1> v2{0};
  assert(static_cast<bool>(v1) == true);
  assert(static_cast<bool>(v2) == false);
  return 0;
}
