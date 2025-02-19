// RUN: %{build} -foffload-fp32-prec-div -foffload-fp32-prec-sqrt -o %t.out
// RUN: %{run} %t.out

// Test if div and sqrt become precise from IEEE-754 perspective when
// -foffload-fp32-prec-div -foffload-fp32-prec-sqrt are passed.

#include <cmath>
#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

constexpr float value = 560.0f;
constexpr float divider = 280.0f;

int32_t ulp_difference(float lhs, float rhs) {
  int32_t lhsInt = *reinterpret_cast<int32_t *>(&lhs);
  int32_t rhsInt = *reinterpret_cast<int32_t *>(&rhs);

  return std::abs(lhsInt - rhsInt);
}

void test_div() {
  sycl::queue q(sycl::default_selector_v);
  float *inValue = (float *)sycl::malloc_shared(sizeof(float), q);
  float *inDivider = (float *)sycl::malloc_shared(sizeof(float), q);
  float *output = (float *)sycl::malloc_shared(sizeof(float), q);
  *inValue = value;
  *inDivider = divider;
  q.submit([&](sycl::handler &h) {
     h.single_task([=] {
       float res = *inValue / *inDivider;
       *output = res;
     });
   }).wait();

  float hostRef = value / divider;
  int ulpDiff = ulp_difference(hostRef, *output);
  assert(std::abs(ulpDiff) < 1 && "Division is not precise");
}

void test_sqrt() {
  sycl::queue q(sycl::default_selector_v);
  float *inValue = (float *)sycl::malloc_shared(sizeof(float), q);
  float *output = (float *)sycl::malloc_shared(sizeof(float), q);
  *inValue = value;
  q.submit([&](sycl::handler &h) {
     h.single_task([=] {
       float res = sycl::sqrt(*inValue);
       *output = res;
     });
   }).wait();

  float hostRef = std::sqrt(value);
  int ulpDiff = ulp_difference(hostRef, *output);
  assert(std::abs(ulpDiff) < 1 && "Sqrt is not precise");
}

int main() {
  test_div();
  test_sqrt();
  return 0;
}
