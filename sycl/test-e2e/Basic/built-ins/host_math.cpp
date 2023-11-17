// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if preview-breaking-changes-supported %{ %{build} -fpreview-breaking-changes -o %t_preview.out %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t_preview.out %}

#include <cmath>
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
  const unsigned int ulpsExpected =
      4; // according to section 4.17.5 of the SYCL 2020 spec, math functions
         // correspond to those from OpenCL 1.2 spec, so this accuracy comes
         // from the OpenCL 1.2 spec
  const double differenceExpected = ulpsExpected * get_ulp_std(reference);
  const double hostDifference = std::fabs(sycl::cospi(value) - reference);

  std::cout << std::setprecision(17) << "cospi: " << '\n'
            << "ref:\t" << reference << '\n'
            << "host:\t" << sycl::cospi(value) << '\n'
            << "diff host:\t " << hostDifference << '\n'
            << "expected:\t" << differenceExpected << std::endl;

  assert(hostDifference <= differenceExpected && "Host result incorrect");
}

void testRemquo() {
  {
    int quo = 0;
    float rem = sycl::remquo(
        86.0f, 10.0f,
        sycl::multi_ptr<int, sycl::access::address_space::global_space>{&quo});
    assert(quo == 9);
    assert(rem == -4);
  }

  {
    int quo = 0;
    float rem = sycl::remquo(
        -10.0, 3.0,
        sycl::multi_ptr<int, sycl::access::address_space::global_space>{&quo});
    assert(quo == -3);
    assert(rem == -1);
  }

  {
    int quo = 0;
    float rem = sycl::remquo(
        0.552879f, 0.219282f,
        sycl::multi_ptr<int, sycl::access::address_space::global_space>{&quo});
    assert(quo == 3);
    assert(rem == -0.10496702790260315f);
  }
}

int main() {
  testCospi();
  testRemquo();
  return 0;
}
