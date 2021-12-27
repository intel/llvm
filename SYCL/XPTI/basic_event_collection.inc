#ifdef XPTI_COLLECTOR

#include "Inputs/test_collector.cpp"

#else

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q{sycl::default_selector{}};

  auto Ptr = sycl::malloc_device<int>(1, Q);

  auto Evt1 = Q.single_task([=]() { Ptr[0] = 1; });

  auto Evt2 = Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Evt1);
    CGH.single_task([=]() { Ptr[0]++; });
  });

  Evt2.wait();

  int Res = 0;
  Q.memcpy(&Res, Ptr, 1);
  Q.wait();

  assert(Res == 2);

  return 0;
}

#endif
