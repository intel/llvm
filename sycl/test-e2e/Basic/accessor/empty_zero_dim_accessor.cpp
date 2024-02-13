// REQUIRES-INTEL-DRIVER: lin: 26690
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests the size and iterator members of an empty zero-dimensional accessor.

#include <sycl/sycl.hpp>

using namespace sycl;

int check_host(bool CheckResult, std::string Msg) {
  if (!CheckResult)
    std::cout << "Case failed: " << Msg << std::endl;
  return !CheckResult;
}

int main() {
  int Failures = 0;
  queue Q;

  buffer<int, 1> EmptyBuf{0};
  assert(EmptyBuf.size() == 0);

  {
    host_accessor<int, 0> EmptyHostAcc{EmptyBuf};
    Failures += check_host(EmptyHostAcc.empty(), "empty() on host_accesor");
    Failures += check_host(EmptyHostAcc.size() == 0, "size() on host_accesor");
    Failures += check_host(EmptyHostAcc.byte_size() == 0,
                           "byte_size() on host_accesor");
    Failures +=
        check_host(EmptyHostAcc.max_size() == 0, "max_size() on host_accesor");
    Failures += check_host(EmptyHostAcc.begin() == EmptyHostAcc.end(),
                           "begin()/end() on host_accesor");
    Failures += check_host(EmptyHostAcc.cbegin() == EmptyHostAcc.cend(),
                           "cbegin()/cend() on host_accesor");
    Failures += check_host(EmptyHostAcc.rbegin() == EmptyHostAcc.rend(),
                           "rbegin()/rend() on host_accesor");
    Failures += check_host(EmptyHostAcc.crbegin() == EmptyHostAcc.crend(),
                           "crbegin()/crend() on host_accesor");
  }

  bool DeviceResults[8] = {false};
  {
    buffer<bool, 1> DeviceResultsBuf{DeviceResults, range<1>{8}};
    Q.submit([&](handler &CGH) {
      accessor<int, 0> EmptyDevAcc{EmptyBuf, CGH};
      accessor DeviceResultsAcc{DeviceResultsBuf, CGH};
      CGH.single_task([=]() {
        DeviceResultsAcc[0] = EmptyDevAcc.empty();
        DeviceResultsAcc[1] = EmptyDevAcc.size() == 0;
        DeviceResultsAcc[2] = EmptyDevAcc.byte_size() == 0;
        DeviceResultsAcc[3] = EmptyDevAcc.max_size() == 0;
        DeviceResultsAcc[4] = EmptyDevAcc.begin() == EmptyDevAcc.end();
        DeviceResultsAcc[5] = EmptyDevAcc.cbegin() == EmptyDevAcc.cend();
        DeviceResultsAcc[6] = EmptyDevAcc.rbegin() == EmptyDevAcc.rend();
        DeviceResultsAcc[7] = EmptyDevAcc.crbegin() == EmptyDevAcc.crend();
      });
    });
  }

  Failures += check_host(DeviceResults[0], "empty() on accessor");
  Failures += check_host(DeviceResults[1], "size() on accessor");
  Failures += check_host(DeviceResults[2], "byte_size() on accessor");
  Failures += check_host(DeviceResults[3], "max_size() on accessor");
  Failures += check_host(DeviceResults[4], "begin()/end() on accessor");
  Failures += check_host(DeviceResults[5], "cbegin()/cend() on accessor");
  Failures += check_host(DeviceResults[6], "rbegin()/rend() on accessor");
  Failures += check_host(DeviceResults[7], "crbegin()/crend() on accessor");

  return Failures;
}
