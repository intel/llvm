// RUN: %clangxx -fsycl -Xclang -verify %s -fsyntax-only
// expected-no-diagnostics

#include <sycl/vector.hpp>

#include <type_traits>

namespace sycl {
namespace detail {
template <typename T, int N> class vec_base_test {
public:
  static void do_check() {
    static_assert(
        std::is_same_v<typename sycl::vec<T, N>::DataType, T[N]>,
        "vec must use plain array as storage, regardless of std::array layout");
  }
};
} // namespace detail
} // namespace sycl

int main() { sycl::detail::vec_base_test<int, 4>::do_check(); }
