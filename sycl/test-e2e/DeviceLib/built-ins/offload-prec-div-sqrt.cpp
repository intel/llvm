// RUN: %{build} -foffload-fp32-prec-div -foffload-fp32-prec-sqrt -o %t.out
// RUN: %{run} %t.out

// Test if div and sqrt become precise from IEEE-754 perspective when
// -foffload-fp32-prec-div -foffload-fp32-prec-sqrt are passed.

#include <cmath>
#include <sycl/ext/oneapi/experimental/builtins.hpp>
#include <sycl/sycl.hpp>

constexpr float value = 560.0f;
constexpr float divider = 280.0f;

// Reference
// https://github.com/KhronosGroup/SYCL-CTS/blob/SYCL-2020/util/accuracy.h
template <typename T> T get_ulp_std(T x) {
  const T inf = std::numeric_limits<T>::infinity();
  const T negative = std::fabs(std::nextafter(x, -inf) - x);
  const T positive = std::fabs(std::nextafter(x, inf) - x);
  return std::fmin(negative, positive);
}

template <typename T> int ulp_difference(const T &lhs, const T &rhs) {
  return get_ulp_std(lhs) - get_ulp_std(rhs);
}

void test_div() {
  sycl::queue q(sycl::default_selector_v, sycl::property::queue::in_order());
  float *in_value = (float *)sycl::malloc_shared(sizeof(float), q);
  float *in_divider = (float *)sycl::malloc_shared(sizeof(float), q);
  float *output = (float *)sycl::malloc_shared(sizeof(float), q);
  *in_value = value;
  *in_divider = divider;
  q.submit([&](sycl::handler &h) {
     h.single_task([=] {
       float res = *in_value / *in_divider;
       *output = res;
     });
   }).wait();

  float hostRef = value / divider;
  int ulpDiff = ulp_difference<float>(hostRef, *output);
  assert(std::abs(ulpDiff) < 1 && "Division is not precise");
}

void test_sqrt() {
  sycl::queue q(sycl::default_selector_v, sycl::property::queue::in_order());
  float *in_value = (float *)sycl::malloc_shared(sizeof(float), q);
  float *output = (float *)sycl::malloc_shared(sizeof(float), q);
  *in_value = value;
  q.submit([&](sycl::handler &h) {
     h.single_task([=] {
       float res = sycl::sqrt(*in_value);
       *output = res;
     });
   }).wait();

  float hostRef = std::sqrt(value);
  int ulpDiff = ulp_difference<float>(hostRef, *output);
  assert(std::abs(ulpDiff) < 1 && "Sqrt is not precise");
}

int main() {
  test_div();
  test_sqrt();
  return 0;
}
