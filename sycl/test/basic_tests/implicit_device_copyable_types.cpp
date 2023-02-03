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
  sycl::queue q;
  {
    std::pair<int, float> pair_arr[5];
    std::pair<int, float> pair;
    q.submit([&](sycl::handler &cgh) {
       cgh.single_task([=]() {
         std::pair<int, float> p0 = pair_arr[0];
         std::pair<int, float> p = pair;
       });
     }).wait_and_throw();
  }

  {
    std::pair<ACopyable, float> pair_arr[5];
    std::pair<ACopyable, float> pair;
    q.submit([&](sycl::handler &cgh) {
       cgh.single_task([=]() {
         std::pair<ACopyable, float> p0 = pair_arr[0];
         std::pair<ACopyable, float> p = pair;
       });
     }).wait_and_throw();
  }

  {
    std::tuple<int, float, bool> tuple_arr[5];
    std::tuple<int, float, bool> tuple;
    q.submit([&](sycl::handler &cgh) {
       cgh.single_task([=]() {
         std::tuple<int, float, bool> t0 = tuple_arr[0];
         std::tuple<int, float, bool> t = tuple;
       });
     }).wait_and_throw();
  }

  {
    std::tuple<ACopyable, float, bool> tuple_arr[5];
    std::tuple<ACopyable, float, bool> tuple;
    q.submit([&](sycl::handler &cgh) {
       cgh.single_task([=]() {
         std::tuple<ACopyable, float, bool> t0 = tuple_arr[0];
         std::tuple<ACopyable, float, bool> t = tuple;
       });
     }).wait_and_throw();
  }

  {
    std::variant<int, float, bool> variant_arr[5];
    std::variant<int, float, bool> variant;
    q.submit([&](sycl::handler &cgh) {
       cgh.single_task([=]() {
         std::variant<int, float, bool> v0 = variant_arr[0];
         std::variant<int, float, bool> v = variant;
       });
     }).wait_and_throw();
  }

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
    std::array<int, 513> arr_arr[5];
    std::array<int, 513> arr;
    q.submit([&](sycl::handler &cgh) {
       cgh.single_task([=]() {
         std::array<int, 513> arr0 = arr_arr[0];
         std::array<int, 513> a = arr;
       });
     }).wait_and_throw();
  }

  {
    std::array<ACopyable, 513> arr_arr[5];
    std::array<ACopyable, 513> arr;
    q.submit([&](sycl::handler &cgh) {
       cgh.single_task([=]() {
         std::array<ACopyable, 513> arr0 = arr_arr[0];
         std::array<ACopyable, 513> a = arr;
       });
     }).wait_and_throw();
  }

  {
    sycl::queue q{};
    std::optional<int> opt_arr[5];
    std::optional<int> opt;
    q.submit([&](sycl::handler &cgh) {
       cgh.single_task([=]() {
         std::optional<int> o0 = opt_arr[0];
         std::optional<int> o = opt;
       });
     }).wait_and_throw();
  }

  {
    sycl::queue q{};
    std::optional<ACopyable> opt_arr[5];
    std::optional<ACopyable> opt;
    q.submit([&](sycl::handler &cgh) {
       cgh.single_task([=]() {
         std::optional<ACopyable> o0 = opt_arr[0];
         std::optional<ACopyable> o = opt;
       });
     }).wait_and_throw();
  }

  {
    std::string_view strv_arr[5];
    std::string_view strv;
    q.submit([&](sycl::handler &cgh) {
       cgh.single_task([=]() {
         std::string_view str0 = strv_arr[0];
         std::string_view str = strv;
       });
     }).wait_and_throw();
  }

#if __cpp_lib_span >= 202002
  {
    std::vector<int> v(5);
    std::span<int> s{v.begin(), 4};
    q.submit([&](sycl::handler &cgh) {
       cgh.single_task([=]() { int x = s[0]; });
     }).wait_and_throw();
  }
#endif

  {
    std::vector<int> v(5);
    sycl::span<int> s{v.data(), 4};
    q.submit([&](sycl::handler &cgh) {
       cgh.single_task([=]() { int x = s[0]; });
     }).wait_and_throw();
  }

    return 0;
}
