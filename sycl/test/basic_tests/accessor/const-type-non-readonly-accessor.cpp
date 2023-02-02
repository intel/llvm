// RUN: %clangxx -fsyntax-only %fsycl-host-only -Xclang -verify -Xclang -verify-ignore-unexpected=error,note %s
// RUN: %clangxx -fsyntax-only -fsycl -fsycl-device-only -Xclang -verify -Xclang -verify-ignore-unexpected=error,note %s

#include <sycl/sycl.hpp>

using namespace sycl;

constexpr size_t dataSize = 1;

int main() {
  const int a[dataSize] = {1};

  try {
    auto defaultQueue = queue{};
    auto bufA = buffer<const int, 1>{a, range{dataSize}};
    defaultQueue.submit([&](handler &cgh) {
      accessor accA{bufA, cgh, read_write};
    // expected-error@sycl/accessor.hpp:* {{A const qualified DataT is only allowed for a read-only accessor}}
    });
  } catch (const exception &e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }
  return 0;
}
