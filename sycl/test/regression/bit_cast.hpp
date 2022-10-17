// The purpose of this test is to check that the following code can be
// successfully compiled

#include <sycl/sycl.hpp>

int main() {
  sycl::half x;
  int16_t a = sycl::bit_cast<int16_t>(x);
  sycl::bit_cast<sycl::half>(a);

  return 0;
}
