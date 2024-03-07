#include <sycl/sycl.hpp>

template <typename T> bool equal(T x, T y) {
  // Maybe should be C++20's std::equality_comparable.
  if constexpr (std::is_scalar_v<T>) {
    return x == y;
  } else {
    for (size_t i = 0; i < x.size(); ++i)
      if (x[i] != y[i])
        return false;

    return true;
  }
}

template <typename FuncTy, typename ExpectedTy,
          typename... ArgTys>
void test(bool CheckDevice, FuncTy F, ExpectedTy Expected, ArgTys... Args) {
  auto R = F(Args...);
  static_assert(std::is_same_v<decltype(Expected), decltype(R)>);
  assert(equal(R, Expected));

  if (!CheckDevice)
    return;

  sycl::buffer<bool, 1> SuccessBuf{1};
  sycl::queue{}.submit([&](sycl::handler &cgh) {
    sycl::accessor Success{SuccessBuf, cgh};
    cgh.single_task([=]() {
      auto R = F(Args...);
      static_assert(std::is_same_v<decltype(Expected), decltype(R)>);
      Success[0] = equal(R, Expected);
    });
  });
  assert(sycl::host_accessor{SuccessBuf}[0]);
}

template <typename FuncTy, typename ExpectedTy, typename... ArgTys>
void test(FuncTy F, ExpectedTy Expected, ArgTys... Args) {
  test(true /*CheckDevice*/, F, Expected, Args...);
}

// MSVC's STL spoils global namespace with math functions, so use explicit
// "sycl::".
#define F(BUILTIN) [](auto... xs) { return sycl::BUILTIN(xs...); }
