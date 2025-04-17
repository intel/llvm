// REQUIRES: aspect-usm_shared_allocations
//
// This test covers a scenario where virtual functions defintion and their uses
// are all split into different translation units.
//
// RUN: %{build} %S/Inputs/call.cpp %S/Inputs/vf.cpp -o %t.out %helper-includes
// RUN: %{run} %t.out

// RUN: %{build} %S/Inputs/call.cpp %S/Inputs/vf.cpp -o %t.out %helper-includes %O0
// RUN: %{run} %t.out

#include "Inputs/declarations.hpp"

#include <iostream>

#include "Inputs/construct.cpp"

int main() try {
  storage_t HostStorage;

  auto asyncHandler = [](sycl::exception_list list) {
    for (auto &e : list)
      std::rethrow_exception(e);
  };

  sycl::queue q(asyncHandler);
  auto *DeviceStorage = sycl::malloc_shared<storage_t>(1, q);

  for (unsigned TestCase = 0; TestCase < 4; ++TestCase) {
    int HostData = 42;
    construct(q, DeviceStorage, TestCase);
    int Result = call(q, DeviceStorage, HostData);

    auto *Ptr =
        HostStorage.construct</* ret type = */ BaseIncrement>(TestCase, 19, 23);
    Ptr->increment(&HostData);

    assert(Result == HostData);
  }

  sycl::free(DeviceStorage, q);

  return 0;
} catch (sycl::exception &e) {
  std::cout << "Unexpected exception was thrown: " << e.what() << std::endl;
  return 1;
}
