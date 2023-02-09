// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out

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

#if COMPILE_ONLY
  sycl::queue q;
  {
    std::variant<ACopyable> variant_arr[5];
    std::variant<ACopyable> variant;
    q.submit([&](sycl::handler &cgh) {
       cgh.single_task([=]() {
         // std::variant with complex types relies on virtual functions, so
         // they cannot be used within sycl kernels
         auto size = sizeof(variant_arr[0]);
         size = sizeof(variant);
       });
     }).wait_and_throw();
  }
  {
    const std::variant<ACopyable> variant_arr[5];
    const std::variant<ACopyable> variant;
    q.submit([&](sycl::handler &cgh) {
       cgh.single_task([=]() {
         // std::variant with complex types relies on virtual functions, so
         // they cannot be used within sycl kernels
         auto size = sizeof(variant_arr[0]);
         size = sizeof(variant);
       });
     }).wait_and_throw();
  }
  {
    volatile std::variant<ACopyable> variant_arr[5];
    volatile std::variant<ACopyable> variant;
    q.submit([&](sycl::handler &cgh) {
       cgh.single_task([=]() {
         // std::variant with complex types relies on virtual functions, so
         // they cannot be used within sycl kernels
         auto size = sizeof(variant_arr[0]);
         size = sizeof(variant);
       });
     }).wait_and_throw();
  }
  {
    const volatile std::variant<ACopyable> variant_arr[5];
    const volatile std::variant<ACopyable> variant;
    q.submit([&](sycl::handler &cgh) {
       cgh.single_task([=]() {
         // std::variant with complex types relies on virtual functions, so
         // they cannot be used within sycl kernels
         auto size = sizeof(variant_arr[0]);
         size = sizeof(variant);
       });
     }).wait_and_throw();
  }
#endif

  return 0;
}
