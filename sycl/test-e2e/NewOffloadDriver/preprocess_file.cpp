// REQUIRES: target-spir
// Test with `--offload-new-driver` that exercises the ability to create
// and consume preprocessed files that will perform full offloading compiles.

// Create the preprocessed file.
// RUN: %{build} --offload-new-driver -E -o %t.ii

// Compile preprocessed file.
// RUN: %clangxx -Wno-error=unused-command-line-argument -fsycl %{sycl_target_opts} --offload-new-driver %t.ii -o %t.out

// RUN: %{run} %t.out

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
