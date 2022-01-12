#include <CL/sycl.hpp>

using namespace sycl;

int main() {
  queue q;
  q.submit([&](handler &cgh) {
    cgh.single_task([=]() {
      ext::oneapi::experimental::printf("String No. %f\n", 1.0f);
      const char *IntFormatString = "String No. %i\n";
      ext::oneapi::experimental::printf(IntFormatString, 2);
      ext::oneapi::experimental::printf(IntFormatString, 3);
    });
  });

  return 0;
}
