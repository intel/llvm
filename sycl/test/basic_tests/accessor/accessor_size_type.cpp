// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -fsyntax-only

#include <sycl/sycl.hpp>
#include <type_traits>

int main() {
  using accessor_size_type = sycl::accessor<int, 3>::size_type;
  using local_accessor_size_type = sycl::local_accessor<int, 3>::size_type;
  using host_accessor_size_type = sycl::host_accessor<int, 3>::size_type;

  static_assert(std::is_same_v<accessor_size_type, std::size_t>);
  static_assert(std::is_same_v<local_accessor_size_type, std::size_t>);
  static_assert(std::is_same_v<host_accessor_size_type, std::size_t>);
}
