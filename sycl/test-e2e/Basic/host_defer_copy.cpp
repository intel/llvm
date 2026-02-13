// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <cstdlib>
#include <iostream>
#include <sycl/detail/core.hpp>

constexpr int N = 10 * 1024 * 1024;

int main() {
  std::vector<int> vec(N, 1);
  const int *const host_address = &vec[0];
  {
    // Create a buffer with a read-only hostData pointer.
    sycl::buffer<int, 1> buf(static_cast<const int *>(vec.data()),
                             sycl::range<1>{N});

    // Assert that the hostData pointer is being reused.
    {
      sycl::host_accessor<int, 1, sycl::access_mode::read> r_acc{buf};
      assert(&r_acc[0] == host_address && "hostData was copied");
    }

    // Assert that creating a writeable accessor copies the data and the
    // hostData pointer is not being reused.
    {
      sycl::host_accessor<int, 1, sycl::access_mode::write> rw_acc{buf};
      assert(&rw_acc[0] != host_address &&
             "writable accessor references read-only hostData");

      rw_acc[0] = 0;
      assert(rw_acc[0] == 0 && "failed to write to accessor");
    }
  }

  // Assert that the vector was never modified (since hostData is read-only).
  assert(vec[0] == 1 && "read-only hostData was modified");

  std::cout << "Test passed!" << std::endl;
  return EXIT_SUCCESS;
}
