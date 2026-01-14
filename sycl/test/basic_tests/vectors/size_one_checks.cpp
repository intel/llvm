// RUN: %clangxx -fsycl %s -o %t_default.out
// RUN: %t_default.out
// RUN: %clangxx -fsycl -D__SYCL_USE_LIBSYCL8_VEC_IMPL=1 %s -o %t_vec.out
// RUN: %t_vec.out

#include <sycl/vector.hpp>

int main() {
  sycl::vec<int, 1> v1{42};
  sycl::vec<int, 1> v2{0};
  assert(static_cast<bool>(v1) == true);
  assert(static_cast<bool>(v2) == false);
  return 0;
}
