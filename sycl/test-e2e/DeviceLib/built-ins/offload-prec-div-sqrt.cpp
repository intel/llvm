// RUN: %{build} -foffload-fp32-prec-div -foffload-fp32-prec-sqrt -o %t.out
// RUN: %{run} %t.out

// Test if div and sqrt become precise from IEEE-754 perspective when
// -foffload-fp32-prec-div -foffload-fp32-prec-sqrt are passed.

#include <cmath>
#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

constexpr float value = 560.0f;
constexpr float divider = 279.9f;

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
  int ulpDist = std::abs(sycl::bit_cast<int32_t>(hostRef) -
                         sycl::bit_cast<int32_t>(*output));
  assert(ulpDist == 0 && "Division is not precise");
  sycl::free(inValue, q);
  sycl::free(inDivider, q);
  sycl::free(output, q);
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
  int ulpDist = std::abs(sycl::bit_cast<int32_t>(hostRef) -
                         sycl::bit_cast<int32_t>(*output));
  assert(ulpDist == 0 && "Sqrt is not precise");
  sycl::free(inValue, q);
  sycl::free(output, q);
}

int main() {
  test_div();
  test_sqrt();
  return 0;
}
