// RUN: %clangxx -fsycl %s -o %t_default.out
// RUN: %t_default.out

#include <sycl/vector.hpp>

int main() {
  sycl::vec<int, 4> v{1, 2, 3, 4};
  auto sw = v.swizzle<1, 2>();
  assert(sw.lo()[0] == 2);
  assert(sw.hi()[0] == 3);

  assert((sw + sw).lo()[0] == 4);

  assert(sw.swizzle<0>()[0] == 2);
  assert(sw.swizzle<1>()[0] == 3);

  {
    auto tmp = sw.swizzle<1, 0>();
    assert(tmp[0] == 3);
    assert(tmp[1] == 2);
  }

  {
    auto tmp = (sw + sw).swizzle<1, 0>();

    assert(tmp[0] == 6);
    assert(tmp[1] == 4);
  }

  return 0;
}
