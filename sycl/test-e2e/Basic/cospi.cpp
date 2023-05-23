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

int main() {
  const double value = 0.4863334355; // some random value
  const double reference =
      0.042921588887841428949787569990803604014217853546142578125; // calculated
                                                                   // with
                                                                   // oclmath
  double res;

  {
    sycl::queue queue;
    sycl::buffer<double, 1> cospi_buf(&res, 1);
    queue
        .submit([&](sycl::handler &cgh) {
          auto cospi_acc = cospi_buf.get_access(cgh);
          cgh.single_task([=]() { cospi_acc[0] = sycl::cospi(value); });
        })
        .wait_and_throw();
  }

  const unsigned int ulpsExpected = 4; // from OpenCL spec
  const double differenceExpected = ulpsExpected * get_ulp_std(reference);
  const double hostDifference = std::fabs(sycl::cospi(value) - reference);
  const double devDifference = std::fabs(res - reference);

  std::cout << std::setprecision(17) << "cospi: " << '\n'
            << "ref:\t" << reference << '\n'
            << "host:\t" << sycl::cospi(value) << '\n'
            << "device:\t" << res << '\n'
            << "diff host:\t " << hostDifference << '\n'
            << "diff device:\t " << devDifference << '\n'
            << "expected:\t" << differenceExpected << std::endl;

  assert(hostDifference <= differenceExpected && "Host result incorrect");
  assert(devDifference <= differenceExpected && "Device result incorrect");

  return 0;
}
