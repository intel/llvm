// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <iomanip>
#include <iostream>
#include <sycl.hpp>

template <typename T> T get_ulp_std(T x) {
  const T inf = std::numeric_limits<T>::infinity();
  const T negative = std::fabs(std::nextafter(x, -inf) - x);
  const T positive = std::fabs(std::nextafter(x, inf) - x);
  return std::fmin(negative, positive);
}

void testCospi() {
  const double value = 0.4863334355; // some random value
  const double reference =
      0.042921588887841428949787569990803604014217853546142578125; // calculated
                                                                   // with
                                                                   // oclmath
  const unsigned int ulpsExpected = 4; // from OpenCL spec
  const double differenceExpected = ulpsExpected * get_ulp_std(reference);
  const double hostDifference = std::fabs(sycl::cospi(value) - reference);

  std::cout << std::setprecision(17) << "cospi: " << '\n'
            << "ref:\t" << reference << '\n'
            << "host:\t" << sycl::cospi(value) << '\n'
            << "diff host:\t " << hostDifference << '\n'
            << "expected:\t" << differenceExpected << std::endl;

  assert(hostDifference <= differenceExpected && "Host result incorrect");
}

int main() {
  testCospi();
  return 0;
}
