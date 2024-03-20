// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if preview-breaking-changes-supported %{ %{build} -fpreview-breaking-changes -o %t_preview.out %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t_preview.out%}

#include <sycl/sycl.hpp>

template <typename... Ts, typename FuncTy> void TestTypes(FuncTy F) {
  (F(Ts{}), ...);
}

int main() {
  sycl::queue q;

  auto Test = [&](auto F, auto Expected, auto... Args) {
#if defined(__GNUC__) || defined(__clang__)
    std::cout << __PRETTY_FUNCTION__ << std::endl;
#endif
    std::tuple ArgsTuple{Args...};
    auto Result = std::apply(F, ArgsTuple);
    static_assert(std::is_same_v<decltype(Expected), decltype(Result)>);
    assert(Expected == Result);

    sycl::buffer<bool, 1> ResultBuf{1};
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor Result{ResultBuf, cgh};
      cgh.single_task([=]() {
        auto R = std::apply(F, ArgsTuple);
        static_assert(std::is_same_v<decltype(Expected), decltype(R)>);
        Result[0] = Expected == R;
      });
    });
    assert(sycl::host_accessor{ResultBuf}[0]);
  };

  auto TestBitSelect = [&](auto type_val) {
    using T = decltype(type_val);
    auto BitSelect = [](auto... xs) { return sycl::bitselect(xs...); };

    static_assert(std::is_integral_v<T>,
                  "Only integer test is implemented here!");
    Test(BitSelect, T{0b0110}, T{0b1100}, T{0b0011}, T{0b1010});
  };

  TestTypes<signed char, unsigned char, char, long, long long, unsigned long,
            unsigned long long>(TestBitSelect);

  auto TestSelect = [&](auto type_val) {
    using T = decltype(type_val);
    auto Select = [](auto... xs) { return sycl::select(xs...); };

    Test(Select, T{0}, T{1}, T{0}, true);
    Test(Select, T{1}, T{1}, T{0}, false);
  };

  TestTypes<signed char, unsigned char, char>(TestSelect);

  return 0;
}
