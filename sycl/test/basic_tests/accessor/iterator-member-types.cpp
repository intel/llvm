// RUN: %clangxx -fsycl -fsyntax-only %s
//
// Purpose of this test is to check that [accessor|host_accessor]::iterator and
// ::const_iterator are aliased to the correct type.
// FIXME: extend this test to also check ::reverse_iterator and
// ::const_reverse_iterator
#include <sycl/sycl.hpp>

#include <type_traits>

template <typename DataT, int Dimensions, sycl::access_mode AccessMode,
          sycl::target AccessTarget = sycl::target::device>
void check_accessor() {
  using AccessorT =
      typename sycl::accessor<DataT, Dimensions, AccessMode, AccessTarget>;
  static_assert(std::is_same_v<sycl::detail::accessor_iterator<
                                   typename AccessorT::value_type, Dimensions>,
                               typename AccessorT::iterator>);

  static_assert(
      std::is_same_v<sycl::detail::accessor_iterator<
                         const typename AccessorT::value_type, Dimensions>,
                     typename AccessorT::const_iterator>);
}

template <typename DataT, int Dimensions, sycl::access_mode AccessMode>
void check_host_accessor() {
  using AccessorT = typename sycl::host_accessor<DataT, Dimensions, AccessMode>;
  static_assert(std::is_same_v<sycl::detail::accessor_iterator<
                                   typename AccessorT::value_type, Dimensions>,
                               typename AccessorT::iterator>);

  static_assert(
      std::is_same_v<sycl::detail::accessor_iterator<
                         const typename AccessorT::value_type, Dimensions>,
                     typename AccessorT::const_iterator>);
}

struct user_defined_t {
  char c;
  float f;
  double d;
  sycl::vec<int, 3> v3;
};

int main() {

  check_accessor<int, 1, sycl::access_mode::read>();
  check_accessor<float, 2, sycl::access_mode::write>();
  check_accessor<user_defined_t, 3, sycl::access_mode::read_write>();

  check_host_accessor<user_defined_t, 1, sycl::access_mode::read>();
  check_host_accessor<int, 2, sycl::access_mode::write>();
  check_host_accessor<float, 3, sycl::access_mode::read_write>();

  return 0;
}
