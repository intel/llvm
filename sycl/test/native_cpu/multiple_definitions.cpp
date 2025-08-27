// check kernel launches from function with multiple definitions work/link
// REQUIRES: native_cpu
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu -DSOURCE1 %s -fno-inline -c -o %t1.o
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu -DSOURCE2 %s -fno-inline -c -o %t2.o
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu %t1.o %t2.o -fno-inline -mllvm -sycl-native-cpu-vecz-width=4 -o %t
// RUN: env ONEAPI_DEVICE_SELECTOR="native_cpu:cpu" %t
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu %t1.o %t2.o -mllvm -sycl-native-cpu-vecz-width=4 -o %t2
// RUN: env ONEAPI_DEVICE_SELECTOR="native_cpu:cpu" %t2

#include <sycl/sycl.hpp>

constexpr unsigned N = 5;
using _Array = int[N];

using namespace ::sycl;

inline int func() {
  queue deviceQueue;
  sycl::range<1> range{N};
  const device dev = deviceQueue.get_device();
  const context ctx = deviceQueue.get_context();
  auto OUT_acc = (_Array *)malloc_shared(sizeof(_Array), dev, ctx);
  for (int i = 0; i < N; i++)
    (*OUT_acc)[i] = 2;

  deviceQueue.submit([&](handler &cgh) {
    cgh.parallel_for<class Test>(
        range, [=](sycl::id<1> ID) { (*OUT_acc)[ID.get(0)] = 1; });
  });

  deviceQueue.wait();

  for (int i = 0; i < N; i++) {
    if ((*OUT_acc)[i] != 1)
      return 1;
  }

  sycl::free(OUT_acc, deviceQueue);
  return 0;
}

#ifdef SOURCE1
int (*fref)() = &func;
#endif
#ifdef SOURCE2
int (*fref2)() = &func;
extern int (*fref)();
int main(void) { return fref() + fref2(); }
#endif // SOURCE2
