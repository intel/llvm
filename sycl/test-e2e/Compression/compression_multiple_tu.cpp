// End-to-End test for testing device image compression when we have two
// translation units, one compressed and one not compressed.
// REQUIRES: zstd

// RUN: %clangxx --offload-compress -DENABLE_KERNEL1 -fsycl -O0 -shared -fPIC %s -o %t_kernel1.so
// RUN: %clangxx -DENABLE_KERNEL2 -fsycl -O0 -shared -fPIC %s -o %t_kernel2.so

// RUN: %clangxx -fsycl %t_kernel1.so %t_kernel2.so %s -Wl,-rpath=%T -o %t_compress.out
// RUN: %{run} %t_compress.out
#if defined(ENABLE_KERNEL1) || defined(ENABLE_KERNEL2)
#include <sycl/detail/core.hpp>

using namespace sycl;

class TestFnObj {
public:
  TestFnObj(buffer<int> &buf, handler &cgh)
      : data(buf.get_access<access::mode::write>(cgh)) {}
  accessor<int, 1, access::mode::write, access::target::device> data;
  void operator()(id<1> item) const { data[item] = item[0]; }
};
#endif

void kernel1();
void kernel2();

#ifdef ENABLE_KERNEL1
void kernel1() {
  static int data[10];
  {
    buffer<int> b(data, range<1>(10));
    queue q;
    q.submit([&](sycl::handler &cgh) {
      TestFnObj kernel(b, cgh);
      cgh.parallel_for(range<1>(10), kernel);
    });
  }
  for (int i = 0; i < 10; i++) {
    assert(data[i] == i);
  }
}
#endif

#ifdef ENABLE_KERNEL2
void kernel2() {
  static int data[256];
  {
    buffer<int> b(data, range<1>(256));
    queue q;
    q.submit([&](handler &cgh) {
      TestFnObj kernel(b, cgh);
      cgh.parallel_for(range<1>(256), kernel);
    });
  }
  for (int i = 0; i < 256; i++) {
    assert(data[i] == i);
  }
}
#endif

#if not defined(ENABLE_KERNEL1) && not defined(ENABLE_KERNEL2)
int main() {
  kernel1();
  kernel2();

  return 0;
}
#endif
