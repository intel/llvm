// RUN: %clangxx -fsycl -Xclang -verify %s -fsyntax-only
// RUN: %clangxx -fsycl -Xclang -verify %s -fsyntax-only -fpreview-breaking-changes
// RUN: %clangxx -fsycl -Xclang -verify %s -fsyntax-only -D__SYCL_USE_PLAIN_ARRAY_AS_VEC_STORAGE=1
// expected-no-diagnostics

#include <sycl/vector.hpp>

#include <type_traits>

namespace sycl {
namespace detail {
template <typename T, int N> class vec_base_test {
public:
  static void do_check() {
    constexpr bool uses_std_array =
        std::is_same_v<typename sycl::vec<T, N>::DataType, std::array<T, N>>;
    constexpr bool uses_plain_array =
        std::is_same_v<typename sycl::vec<T, N>::DataType, T[N]>;

    constexpr bool std_array_and_plain_array_have_the_same_layout =
        sizeof(std::array<T, N>) == sizeof(T[N]) &&
        alignof(std::array<T, N>) == alignof(T[N]);

#if defined(__INTEL_PREVIEW_BREAKING_CHANGES) ||                               \
    __SYCL_USE_PLAIN_ARRAY_AS_VEC_STORAGE
    static_assert(uses_plain_array,
                  "We must use plain array regardless of "
                  "layout, because user is opted-in for a potential ABI-break");
#else
    static_assert(std_array_and_plain_array_have_the_same_layout ==
                      uses_plain_array,
                  "If layouts are the same, we must use safer plain array "
                  "instead of std::array, or vice versa");
    static_assert(
        !std_array_and_plain_array_have_the_same_layout == uses_std_array,
        "If layouts are not the same, we must use std::array to preserve ABI");
#endif
  }
};
} // namespace detail
} // namespace sycl

int main() { sycl::detail::vec_base_test<int, 4>::do_check(); }
