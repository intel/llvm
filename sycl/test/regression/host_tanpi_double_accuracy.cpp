// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out
//
// Checks that sycl::tanpi with double has the accuracy required by the
// specification.

#include <cmath>
#include <iomanip>
#include <sycl/sycl.hpp>

double get_ulp(double X) {
  double Inf = std::numeric_limits<double>::infinity();
  double Negative = std::fabs(std::nextafter(X, -Inf) - X);
  double Positive = std::fabs(std::nextafter(X, Inf) - X);
  return std::fmin(Negative, Positive);
}

// NOTE: Acc is the expected accuracy in ULPs. For non-half builtins, the
// SYCL 2020 specification specifies that it should follow the accuracy required
// by the OpenCL specification, which for tanpi means <= 6 ULP. See
// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_C.html#relative-error-as-ulps.
int check_tanpi(double input, double ReferenceRes, unsigned int Acc = 6) {
  double Value = sycl::tanpi(input);

  double Difference = std::fabs(Value - ReferenceRes);
  double DifferenceExpected = Acc * get_ulp(ReferenceRes);

  if (Difference > DifferenceExpected) {
    std::cout << "sycl::tanpi(" << input << "): " << Value
              << " is not the required accuracy compared to " << ReferenceRes
              << std::endl;
    return 1;
  }
  return 0;
}

int main() {
  int Failures = 0;
  std::cout << std::setprecision(64);
  // Reference results are taken from SYCL-CTS.
  Failures += check_tanpi(0.5090197771,
                          -35.2807704551884313559639849700033664703369140625);
  Failures += check_tanpi(0.4812775633,
                          16.98190960997280996025438071228563785552978515625);
  Failures += check_tanpi(0.5037494847,
                          -84.8903754371682879309446434490382671356201171875);
  Failures += check_tanpi(0.4948622932,
                          61.95025450742041783769309404306113719940185546875);
  Failures += check_tanpi(0.506352514,
                          -50.10105070167288232596547459252178668975830078125);
  return Failures;
}
