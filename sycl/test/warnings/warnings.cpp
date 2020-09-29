// RUN: %clangxx -fsycl --no-system-header-prefix=CL/sycl -fsyntax-only -Wall -Wextra -Wno-ignored-attributes -Wno-deprecated-declarations -Wpessimizing-move -Wunused-variable -Wmismatched-tags -Wunneeded-internal-declaration -Werror -Wno-unknown-cuda-version %s -o %t.out

#include <CL/sycl.hpp>

using namespace cl::sycl;
int main() {
  vec<long, 4> newVec;
  queue myQueue;
  buffer<vec<long, 4>, 1> resultBuf{&newVec, range<1>{1}};
  myQueue.submit([&](handler &cgh) {
    auto writeResult = resultBuf.get_access<access::mode::write>(cgh);
    cgh.single_task<class kernel_name>([=]() {
      writeResult[0] = (vec<int, 4>{1, 2, 3, 4}).template convert<long>();
    });
  });
  return 0;
}
