// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <cassert>
#include <cmath>
#include <limits>
#include <sycl/builtins.hpp>
#include <sycl/detail/core.hpp>

template <typename T> struct nextafter_test {
  sycl::accessor<T, 1, sycl::access_mode::read> deviceInputs;
  sycl::accessor<T, 1, sycl::access_mode::write> deviceOutputs;
  void operator()() const {
    for (size_t i{0}; i < deviceOutputs.size(); ++i) {
      deviceOutputs[i] =
          sycl::nextafter(deviceInputs[2 * i], deviceInputs[2 * i + 1]);
    }
  }
};

struct dfh_tuple {
  double d;
  float f;
  sycl::half h;
  template <typename T> T get() const {
    if constexpr (std::is_same_v<T, double>)
      return d;
    if constexpr (std::is_same_v<T, float>)
      return f;
    if constexpr (std::is_same_v<T, sycl::half>)
      return h;
    return T{};
  }
};

template <typename T> void run_nextafter_test(sycl::queue &deviceQueue) {
  constexpr static size_t NumTests{11};
  constexpr static T nan{std::numeric_limits<T>::quiet_NaN()};
  std::array<T, 2 * NumTests> inputs{
      // positive zeros
      0.0,
      0.0,
      // negative zeros
      -0.0,
      -0.0,
      // neg and pos zero
      -0.0,
      0.0,
      // pos and neg zero
      0.0,
      -0.0,
      // positive away from zero
      1.0,
      2.0,
      // positive towards zero
      1.0,
      0.0,
      // negative towards zero
      -1.0,
      0.0,
      // negative away from zero
      -1.0,
      -2.0,
      // NaN first arg
      nan,
      1.0,
      // NaN second arg
      1.0,
      nan,
      // NaN both args
      nan,
      nan,
  };

  std::array<T, NumTests> outputs{};

  std::array<T, NumTests> expected{
      0.0,
      -0.0,
      0.0,
      -0.0,
      dfh_tuple{1.0000000000000002, 1.0000001f, 1.001f}.get<T>(),
      dfh_tuple{0.99999999999999988897769753748434595763683319091796875,
                0.999999940395355224609375f, 0.99951171875f}
          .get<T>(),
      dfh_tuple{-0.99999999999999988897769753748434595763683319091796875,
                -0.999999940395355224609375f, -0.99951171875f}
          .get<T>(),
      dfh_tuple{-1.0000000000000002, -1.0000001f, -1.001f}.get<T>(),
      nan,
      nan,
      nan,
  };

  {
    sycl::buffer inputBuffer{inputs};
    sycl::buffer outputBuffer{outputs};
    deviceQueue
        .submit([&inputBuffer, &outputBuffer](sycl::handler &cgh) {
          sycl::accessor inputAcc{inputBuffer, cgh, sycl::read_only};
          sycl::accessor outputAcc{outputBuffer, cgh, sycl::write_only};
          cgh.single_task(nextafter_test<T>{inputAcc, outputAcc});
        })
        .wait_and_throw();
  }

  for (size_t i{0}; i < NumTests; ++i) {
    if (sycl::isnan(expected[i]))
      assert(sycl::isnan(outputs[i]));
    else
      assert(outputs[i] == expected[i] &&
             std::signbit(outputs[i]) == std::signbit(expected[i]));
  }
}

int main() {
  sycl::queue deviceQueue{};
  run_nextafter_test<float>(deviceQueue);
  if (deviceQueue.get_device().has(sycl::aspect::fp64))
    run_nextafter_test<double>(deviceQueue);
  if (deviceQueue.get_device().has(sycl::aspect::fp16))
    run_nextafter_test<sycl::half>(deviceQueue);
  return 0;
}
