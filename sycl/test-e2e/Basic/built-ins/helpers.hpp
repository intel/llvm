#include <sycl/detail/core.hpp>

#include <sycl/builtins.hpp>

template <typename T, typename D> bool equal(T x, T y, D delta) {
  // Maybe should be C++20's std::equality_comparable.
  auto Abs = [](auto x) {
    if constexpr (std::is_unsigned_v<decltype(x)>)
      return x;
    else
      return std::abs(x);
  };
  if constexpr (std::is_scalar_v<T> || std::is_same_v<sycl::half, T>) {
    return Abs(x - y) <= delta;
  } else {
    for (size_t i = 0; i < x.size(); ++i)
      if (Abs(x[i] - y[i]) > delta)
        return false;

    return true;
  }
}

template <typename FuncTy, typename ExpectedTy, typename... ArgTys>
void test(bool CheckDevice, double delta, FuncTy F, ExpectedTy Expected,
          ArgTys... Args) {
  auto R = F(Args...);
  static_assert(std::is_same_v<decltype(Expected), decltype(R)>);
  assert(equal(R, Expected, delta));

  if (!CheckDevice)
    return;

  sycl::buffer<bool, 1> SuccessBuf{1};

  // Make sure we don't use fp64 on devices that don't support it.
  sycl::detail::get_elem_type_t<ExpectedTy> d(delta);

  sycl::queue{}.submit([&](sycl::handler &cgh) {
    sycl::accessor Success{SuccessBuf, cgh};
    cgh.single_task([=]() {
      auto R = F(Args...);
      static_assert(std::is_same_v<decltype(Expected), decltype(R)>);
      Success[0] = equal(R, Expected, d);
    });
  });
  assert(sycl::host_accessor{SuccessBuf}[0]);
}

template <typename FuncTy, typename ExpectedTy, typename... ArgTys>
void test(FuncTy F, ExpectedTy Expected, ArgTys... Args) {
  test(true /*CheckDevice*/, 0.0 /*delta*/, F, Expected, Args...);
}
template <typename FuncTy, typename ExpectedTy,
          typename... ArgTys>
void test(bool CheckDevice, FuncTy F, ExpectedTy Expected, ArgTys... Args) {
  test(CheckDevice, 0.0 /*delta*/, F, Expected, Args...);
}
template <typename FuncTy, typename ExpectedTy, typename... ArgTys>
void test(double delta, FuncTy F, ExpectedTy Expected, ArgTys... Args) {
  test(true /*CheckDevice*/, delta, F, Expected, Args...);
}

// MSVC's STL spoils global namespace with math functions, so use explicit
// "sycl::".
#define F(BUILTIN) [](auto... xs) { return sycl::BUILTIN(xs...); }
