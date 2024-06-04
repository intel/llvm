// REQUIRES: opencl, cpu
// RUN: %{build} -o %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="opencl:*" %{run-unfiltered-devices} %t.out 1 &> tmp.txt
// RUN: cat tmp.txt | env ONEAPI_DEVICE_SELECTOR="opencl:*,cpu" %{run-unfiltered-devices} %t.out
// RUN: cat tmp.txt | env ONEAPI_DEVICE_SELECTOR="opencl:cpu,cpu" %{run-unfiltered-devices} %t.out

// on the first run we pass a dummy arg to the app. On seeing that, we count the
// number of CPU devices and output it. That is piped  to a file. On subsequent
// runs we cat the file and pipe that to app. The app then compares the number
// of CPU devices to that number, erroring if they differ.

// clang++ -fsycl -o ndd.bin no_duplicate_devices.cpp

#include <string>
#include <sycl/detail/core.hpp>

using namespace sycl;

int main(int argc, char *argv[]) {

  unsigned numCPUDevices = device::get_devices(info::device_type::cpu).size();

  // when arg is provided, simply count the number of CPU devices and output.
  if (argc > 1) {
    std::cout << numCPUDevices << std::endl;
    return 0; // done!
  }

  // when arg is not provided, check the count against the piped value.
  char charBuff[16];
  std::cin.read(charBuff, 16);
  unsigned expectedNumCPUDevices = std::stoi(charBuff);

  assert(numCPUDevices == expectedNumCPUDevices &&
         "wrong number of cpu devices. duplicates.");

  return 0;
}
