// REQUIRES: accelerator

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out | FileCheck %s

// Test that usage of blocking pipes and stream object in the same kernel
// doesn't cause program hang.

// CHECK: 0
// CHECK-NEXT: 1
// CHECK-NEXT: 2
// CHECK-NEXT: 3
// CHECK-NEXT: 4
// CHECK-NEXT: 5
// CHECK-NEXT: 6
// CHECK-NEXT: 7
// CHECK-NEXT: 8
// CHECK-NEXT: 9
// CHECK-NEXT: 0
// CHECK-NEXT: 1
// CHECK-NEXT: 2
// CHECK-NEXT: 3
// CHECK-NEXT: 4
// CHECK-NEXT: 5
// CHECK-NEXT: 6
// CHECK-NEXT: 7
// CHECK-NEXT: 8
// CHECK-NEXT: 9

#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/pipes.hpp>
#include <sycl/stream.hpp>

using namespace sycl;

constexpr size_t N = 10;

// Specialize a pipe type
using my_pipe = pipe<class some_pipe, int>;

class producer_task;
class consumer_task;

void producer(queue &q) {
  // Launch the producer kernel
  q.submit([&](handler &cgh) {
    // Get read access to src buffer
    sycl::stream out(1024, 16, cgh);
    cgh.single_task<producer_task>([=] {
      for (int i = 0; i < N; i++) {
        // Blocking write an int to the pipe
        my_pipe::write(i);
        out << i << sycl::endl;
      }
    });
  });
}

void consumer(queue &q) {
  // Launch the consumer kernel
  q.submit([&](handler &cgh) {
    // Get write access to dst buffer
    sycl::stream out(1024, 16, cgh);
    cgh.single_task<consumer_task>([=] {
      for (int i = 0; i < N; i++) {
        // Blocking read an int from the pipe
        int tmp = my_pipe::read();
        out << tmp << sycl::endl;
      }
    });
  });
}

int main() {
  {
    queue q;

    producer(q);
    consumer(q);
    q.wait();
  }

  return 0;
}
