// RUN: %clangxx -fsycl %s

#include <sycl/sycl.hpp>

template <typename T, typename ExpectedBaseType> void Check() {
  static_assert(std::is_base_of_v<ExpectedBaseType, sycl::is_group<T>>);
  static_assert(sycl::is_group<T>::value == ExpectedBaseType::value);
  static_assert(sycl::is_group_v<T> == ExpectedBaseType::value);
}

int main() {
  Check<sycl::group<1>, std::true_type>();
  Check<sycl::group<2>, std::true_type>();
  Check<sycl::group<3>, std::true_type>();
  Check<sycl::sub_group, std::true_type>();

  Check<int, std::false_type>();
  Check<sycl::queue, std::false_type>();
  Check<sycl::device, std::false_type>();

  return 0;
}
