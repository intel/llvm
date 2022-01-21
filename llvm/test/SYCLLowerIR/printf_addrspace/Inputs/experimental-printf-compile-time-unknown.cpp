#include <CL/sycl.hpp>

using namespace sycl;

int foo(int k) {
  queue q;
  buffer<int, 1> buf(k);
  q.submit([&](handler &cgh) {
    auto acc = buf.get_access(cgh);
    cgh.single_task([=]() {
      if (acc[0] == 0)
        ext::oneapi::experimental::printf("String 0\n");
      else
        ext::oneapi::experimental::printf("String 1\n");
    });
  });
  return 0;
}
