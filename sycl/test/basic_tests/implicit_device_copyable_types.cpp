// RUN: %clangxx -fsycl -fsyntax-only %s
// RUN: %if preview-breaking-changes-supported %{ %clangxx -fsycl -fpreview-breaking-changes -fsyntax-only %s  %}

#include <sycl/sycl.hpp>
#include <variant>

struct ACopyable {
  int i;
  ACopyable() = default;
  ACopyable(int _i) : i(_i) {}
  ACopyable(const ACopyable &x) : i(x.i) {}
};

template <> struct sycl::is_device_copyable<ACopyable> : std::true_type {};

int main() {
  static_assert(sycl::is_device_copyable_v<std::pair<int, float>>);
  static_assert(sycl::is_device_copyable_v<std::pair<ACopyable, float>>);
  static_assert(sycl::is_device_copyable_v<std::tuple<int, float, bool>>);
  static_assert(sycl::is_device_copyable_v<std::tuple<ACopyable, float, bool>>);
  static_assert(sycl::is_device_copyable_v<std::variant<int, float, bool>>);
  static_assert(sycl::is_device_copyable_v<std::variant<ACopyable>>);
  static_assert(sycl::is_device_copyable_v<std::array<int, 513>>);
  static_assert(sycl::is_device_copyable_v<std::array<ACopyable, 513>>);
  static_assert(sycl::is_device_copyable_v<std::optional<int>>);
  static_assert(sycl::is_device_copyable_v<std::optional<ACopyable>>);
  static_assert(sycl::is_device_copyable_v<std::string_view>);
#if __cpp_lib_span >= 202002
  static_assert(sycl::is_device_copyable_v<std::span<int>>);
#endif
  static_assert(sycl::is_device_copyable_v<const sycl::span<int>>);

  // const
  static_assert(sycl::is_device_copyable_v<const std::pair<int, float>>);
  static_assert(sycl::is_device_copyable_v<const std::pair<ACopyable, float>>);
  static_assert(sycl::is_device_copyable_v<const std::tuple<int, float, bool>>);
  static_assert(
      sycl::is_device_copyable_v<const std::tuple<ACopyable, float, bool>>);
  static_assert(
      sycl::is_device_copyable_v<const std::variant<int, float, bool>>);
  static_assert(sycl::is_device_copyable_v<const std::variant<ACopyable>>);
  static_assert(sycl::is_device_copyable_v<const std::array<int, 513>>);
  static_assert(sycl::is_device_copyable_v<const std::array<ACopyable, 513>>);
  static_assert(sycl::is_device_copyable_v<const std::optional<int>>);
  static_assert(sycl::is_device_copyable_v<const std::optional<ACopyable>>);
  static_assert(sycl::is_device_copyable_v<const std::string_view>);
#if __cpp_lib_span >= 202002
  static_assert(sycl::is_device_copyable_v<const std::span<int>>);
#endif
  static_assert(sycl::is_device_copyable_v<const sycl::span<int>>);

  // volatile
  static_assert(sycl::is_device_copyable_v<volatile std::pair<int, float>>);
  static_assert(
      sycl::is_device_copyable_v<volatile std::pair<ACopyable, float>>);
  static_assert(
      sycl::is_device_copyable_v<volatile std::tuple<int, float, bool>>);
  static_assert(
      sycl::is_device_copyable_v<volatile std::tuple<ACopyable, float, bool>>);
  static_assert(
      sycl::is_device_copyable_v<volatile std::variant<int, float, bool>>);
  static_assert(sycl::is_device_copyable_v<volatile std::variant<ACopyable>>);
  static_assert(sycl::is_device_copyable_v<volatile std::array<int, 513>>);
  static_assert(
      sycl::is_device_copyable_v<volatile std::array<ACopyable, 513>>);
  static_assert(sycl::is_device_copyable_v<volatile std::optional<int>>);
  static_assert(sycl::is_device_copyable_v<volatile std::optional<ACopyable>>);
  static_assert(sycl::is_device_copyable_v<volatile std::string_view>);
#if __cpp_lib_span >= 202002
  static_assert(sycl::is_device_copyable_v<volatile std::span<int>>);
#endif
  static_assert(sycl::is_device_copyable_v<volatile sycl::span<int>>);

  // const volatile
  static_assert(
      sycl::is_device_copyable_v<const volatile std::pair<int, float>>);
  static_assert(
      sycl::is_device_copyable_v<const volatile std::pair<ACopyable, float>>);
  static_assert(
      sycl::is_device_copyable_v<const volatile std::tuple<int, float, bool>>);
  static_assert(sycl::is_device_copyable_v<
                const volatile std::tuple<ACopyable, float, bool>>);
  static_assert(sycl::is_device_copyable_v<
                const volatile std::variant<int, float, bool>>);
  static_assert(
      sycl::is_device_copyable_v<const volatile std::variant<ACopyable>>);
  static_assert(
      sycl::is_device_copyable_v<const volatile std::array<int, 513>>);
  static_assert(
      sycl::is_device_copyable_v<const volatile std::array<ACopyable, 513>>);
  static_assert(sycl::is_device_copyable_v<const volatile std::optional<int>>);
  static_assert(
      sycl::is_device_copyable_v<const volatile std::optional<ACopyable>>);
  static_assert(sycl::is_device_copyable_v<const volatile std::string_view>);
#if __cpp_lib_span >= 202002
  static_assert(sycl::is_device_copyable_v<const volatile std::span<int>>);
#endif
  static_assert(sycl::is_device_copyable_v<const volatile sycl::span<int>>);

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  // Extra checks
  static_assert(sycl::is_device_copyable_v<sycl::vec<int, 4>>);
  static_assert(sycl::is_device_copyable_v<sycl::vec<sycl::half, 4>>);
  static_assert(
      sycl::is_device_copyable_v<sycl::vec<sycl::ext::oneapi::bfloat16, 4>>);

  struct S {
    sycl::vec<int, 4> v;
  };
  static_assert(sycl::is_device_copyable_v<S>);
#endif

  return 0;
}
