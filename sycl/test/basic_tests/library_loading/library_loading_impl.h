#include <CL/sycl.hpp>

using namespace cl::sycl;

int main() {
  queue q;
  q.submit([&](handler &cgh) {});
}
