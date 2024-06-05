// REQUIRES: cpu
// UNSUPPORTED: windows
//   DeferredMemory Destruction not presently supported on Windows.

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

using namespace sycl;
using namespace sycl::access;

static constexpr size_t BUFFER_SIZE = 256;

void test() {
  queue Q;
  std::unique_ptr<host_accessor<int>> HostAcc;
  {
    buffer<int, 1> Buffer{BUFFER_SIZE};
    HostAcc.reset(new host_accessor(Buffer));
    // Host accessor should block kernel execution
    Q.submit([&](handler &CGH) {
      auto Acc = Buffer.template get_access<mode::write>(CGH);
      CGH.parallel_for<class Test>(BUFFER_SIZE,
                                   [=](item<1> Id) { Acc[Id] = 0; });
    });
    // Buffer destructor should not wait till all operations completion here
  }
  // Unblock kernel execution
  HostAcc.reset(nullptr);
  Q.wait();
}

int main() {
  test();
  return 0;
}
