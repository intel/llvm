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
  static_assert(sycl::is_device_copyable_v<sycl::span<int>>);

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
#endif

  return 0;
}
