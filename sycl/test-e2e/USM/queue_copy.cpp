// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if level_zero && linux %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{run} %t.out %}
//

#include <cassert>
#include <cstddef>
#include <iostream>
#include <numeric>

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

// This test checks that queue USM functions are properly waited
// when using immediate queue

int main() {

  const size_t Size = 1024;
  queue Queue{};
  device Dev = Queue.get_device();

  if (!(Dev.get_info<info::device::usm_device_allocations>()))
    return 0;

  using T = int;
  std::vector<T> DataA(Size), DataB(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);

  T *Ptr = malloc_device<T>(Size, Queue);

  Queue.copy(DataA.data(), Ptr, Size);
  Queue.wait_and_throw();

  Queue.copy(Ptr, DataB.data(), Size);
  Queue.wait_and_throw();

  free(Ptr, Queue);

  for (size_t i = 0; i < Size; i++) {
    assert(DataA[i] == DataB[i]);
  }

  std::cout << "Passed\n";
  return 0;
}
