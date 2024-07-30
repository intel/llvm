// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out
// RUN: %if preview-breaking-changes-supported %{ %clangxx -fsycl -fpreview-breaking-changes %s -o %t_preview.out %}
// RUN: %if preview-breaking-changes-supported %{ %t_preview.out %}

#include <sycl/vector.hpp>

int main() {
  {
    sycl::vec<int, 4> a{1, 2, 3, 4};
    sycl::vec<int, 4> b{4, 3, 2, 1};
    auto sw_a = a.swizzle<0>();
    auto sw_b = b.swizzle<0>();

    // Make sure copy-assignment is available and is doing the right thing.
    // Template overloads of operator= aren't selected for the assignment from
    // the same swizzle type.
    static_assert(std::is_same_v<decltype(sw_a), decltype(sw_b)>);
    sw_a = sw_b;
    assert(a[0] == 4);

    // Different type, just for the test completeness.
    auto sw_c = b.swizzle<1>();
    static_assert(!std::is_same_v<decltype(sw_a), decltype(sw_c)>);
    sw_a = sw_c;
    assert(a[0] == 3);
  }
  return 0;
}
