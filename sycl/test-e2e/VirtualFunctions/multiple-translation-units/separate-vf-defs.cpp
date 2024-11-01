// REQUIRES: aspect-usm_shared_allocations
//
// We attach calls-indirectly attribute (and therefore device image property)
// to construct kernels at compile step. At that stage we may not see virtual
// function definitions and therefore we won't mark construct kernel as using
// virtual functions and link operation at runtime will fail due to undefined
// references to virtual functions from vtable.
// XFAIL: *
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/15071
//
// This test covers a scenario where virtual functions defintion and their uses
// are split into different translation units. In particular:
// - use and construct kernesl are in the same translation unit
// - but virtual functions are defined in a separate translation unit
//
// RUN: %{build} %S/Inputs/vf.cpp -o %t.out %helper-includes
// RUN: %{run} %t.out

#include "Inputs/declarations.hpp"

#include <iostream>

#include "Inputs/call.cpp"
#include "Inputs/construct.cpp"

int main() try {
  storage_t HostStorage;

  auto asyncHandler = [](sycl::exception_list list) {
    for (auto &e : list)
      std::rethrow_exception(e);
  };

  sycl::queue q(asyncHandler);
  auto *DeviceStorage = sycl::malloc_shared<storage_t>(1, q);

  constexpr oneapi::properties props{oneapi::assume_indirect_calls};
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
