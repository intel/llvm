// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <sycl/vector.hpp>

int main() {
  sycl::vec<int, 1> v1{42};
  sycl::vec<int, 1> v2{0};
  assert(static_cast<bool>(v1) == true);
  assert(static_cast<bool>(v2) == false);
  return 0;
}
