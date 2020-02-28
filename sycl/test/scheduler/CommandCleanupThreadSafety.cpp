// UNSUPPORTED: windows
// RUN: %clangxx -fsycl %s -o %t.out -lpthread
// RUN: %CPU_RUN_PLACEHOLDER %t.out
#include <CL/sycl.hpp>

#include <cassert>
#include <cstddef>
#include <thread>
#include <vector>

// This test checks that the command graph cleanup works properly when
// invoked from multiple threads.
using namespace cl::sycl;

class Foo;

event submitTask(queue &Q, buffer<int, 1> &Buf) {
  return Q.submit([&](handler &Cgh) {
    auto Acc = Buf.get_access<access::mode::read_write>(Cgh);
    Cgh.single_task<Foo>([=]() { Acc[0] = 42; });
  });
}

int main() {
  queue Q;
  buffer<int, 1> Buf(range<1>(1));

  // Create multiple commands, each one dependent on the previous
  std::vector<event> Events;
  const std::size_t NTasks = 16;
  for (std::size_t I = 0; I < NTasks; ++I)
    Events.push_back(submitTask(Q, Buf));

  // Initiate cleanup from multiple threads
  std::vector<std::thread> Threads;
  for (event &E : Events)
    Threads.emplace_back([&]() { E.wait(); });
  for (std::thread &T : Threads)
    T.join();
}
