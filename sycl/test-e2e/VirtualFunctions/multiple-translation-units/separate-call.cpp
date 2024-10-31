// REQUIRES: aspect-usm_shared_allocations
//
// VTables are global variables with possibly external linkage and that causes
// them to be copied into every module we produce during device code split
// which in turn leads to multiple definitions error at runtime.
// XFAIL: *
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/15069
//
// This test covers a scenario where virtual functions defintion and their uses
// are split into different translation units. In particular:
// - both virtual functions and construct kernel are in the same translation
//   unit
// - but use kernel is outlined into a separate translation unit
//
// RUN: %{build} %S/Inputs/call.cpp -o %t.out %helper-includes
// RUN: %{run} %t.out

#include "Inputs/declarations.hpp"

#include <iostream>

#include "Inputs/construct.cpp"
#include "Inputs/vf.cpp"

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
