// RUN: not %clangxx -fsycl -fsycl-device-only -fsyntax-only \
// RUN: %s -I %sycl_include 2>&1 | FileCheck %s

#include <sycl/sycl.hpp>

using namespace sycl;

constexpr size_t dataSize = 1;

int main() {
  const int a[dataSize] = {1};

  try {
    auto defaultQueue = queue{};
    auto bufA = buffer<const int, 1>{a, range{dataSize}};
    defaultQueue.submit([&](handler &cgh) {
      sycl::accessor accA{bufA, cgh, read_write};
    });
  } catch (const exception &e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }
  return 0;
}

// CHECK: static assertion failed due to requirement '!isConst || IsAccessReadOnly': A const qualified DataT is only allowed for a read-only accessor
