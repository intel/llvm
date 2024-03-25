// RUN: %clangxx -fsycl -fsyntax-only %s

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;
  Q.single_task([]() {
    sycl::vec<int, 4> X{1};
    static_assert(std::is_same_v<decltype(X.swizzle<0>())::element_type, int>);
    static_assert(std::is_same_v<decltype(X.swizzle<0>())::value_type, int>);
#ifdef __SYCL_DEVICE_ONLY__
    static_assert(std::is_same_v<decltype(X.swizzle<0>())::vector_t,
                                 sycl::vec<int, 1>::vector_t>);
#endif // __SYCL_DEVICE_ONLY__
  });
  return 0;
}
