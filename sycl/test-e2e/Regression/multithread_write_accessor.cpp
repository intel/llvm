// RUN: %{build} -o %t.out %threads_lib
// RUN: %{run} %t.out
#include <sycl/detail/core.hpp>

#include <cassert>
#include <thread>
#include <vector>

constexpr int NThreads = 8;

class KernelA;

void threadFunction(sycl::buffer<int, 1> &Buf) {
  sycl::queue Q;
  Q.submit([&](sycl::handler &Cgh) {
    auto Acc = Buf.get_access<sycl::access::mode::read_write>(Cgh);
    Cgh.single_task<class KernelA>([=]() { Acc[0] += 1; });
  });
}
int main() {
  std::vector<std::thread> Threads;
  Threads.reserve(NThreads);

  int Val = 0;
  {
    sycl::buffer<int, 1> Buf(&Val, sycl::range<1>(1));
    sycl::queue Q;

    for (int I = 0; I < NThreads; ++I)
      Threads.emplace_back(threadFunction, std::ref(Buf));
    for (auto &t : Threads)
      t.join();
  }
  assert(Val == NThreads);
}
