// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl -fsycl-device-obj=spirv -c -o %t.o %s
// RUN: %clangxx -fsycl -o %t.out %t.o
// RUN: %{run} %t.out

// This test verifies SPIR-V based fat objects.

#include <sycl/detail/core.hpp>

int main() {
  sycl::buffer<size_t, 1> Buffer(4);

  sycl::queue Queue;

  sycl::range<1> NumOfWorkItems{Buffer.size()};

  Queue.submit([&](sycl::handler &cgh) {
    sycl::accessor Accessor{Buffer, cgh, sycl::write_only};
    cgh.parallel_for<class FillBuffer>(NumOfWorkItems, [=](sycl::id<1> WIid) {
      Accessor[WIid] = WIid.get(0);
    });
  });

  sycl::host_accessor HostAccessor{Buffer, sycl::read_only};

  bool MismatchFound = false;
  for (size_t I = 0; I < Buffer.size(); ++I) {
    if (HostAccessor[I] != I) {
      std::cout << "The result is incorrect for element: " << I
                << " , expected: " << I << " , got: " << HostAccessor[I]
                << std::endl;
      MismatchFound = true;
    }
  }

  return MismatchFound;
}
