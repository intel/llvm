// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if preview-breaking-changes-supported %{ %{build} -fpreview-breaking-changes -o %t_preview.out %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t_preview.out%}

#include <sycl/sycl.hpp>

int main() {
  using namespace sycl;
  queue q;

  auto Test = [&](auto F, auto Expected, auto... Args) {
    std::tuple ArgsTuple{Args...};
    auto Result = std::apply(F, ArgsTuple);
    static_assert(std::is_same_v<decltype(Expected), decltype(Result)>);

    auto Equal = [](auto x, auto y) {
      for (size_t i = 0; i < x.size(); ++i)
        if (x[i] != y[i])
          return false;

      return true;
    };

    assert(Equal(Result, Expected));

    buffer<bool, 1> ResultBuf{1};
    q.submit([&](handler &cgh) {
      accessor Result{ResultBuf, cgh};
      cgh.single_task([=]() {
        auto R = std::apply(F, ArgsTuple);
        static_assert(std::is_same_v<decltype(Expected), decltype(R)>);
        Result[0] = Equal(R, Expected);
      });
    });
    assert(host_accessor{ResultBuf}[0]);
  };

  {
    // Test upsample:
    auto Upsample = [](auto... xs) { return upsample(xs...); };
    Test(Upsample, marray<int16_t, 2>{0x203, 0x302}, marray<int8_t, 2>{2, 3},
         marray<uint8_t, 2>{3, 2});
    Test(Upsample, marray<uint16_t, 2>{0x203, 0x302}, marray<uint8_t, 2>{2, 3},
         marray<uint8_t, 2>{3, 2});
  }

  return 0;
}
