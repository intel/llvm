// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using ZE_DEBUG
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Fail that needs investigation
// XFAIL: *

// This test creates a temporary buffer which is used in kernels, but
// destroyed before finalization and execution of the graph.

#include "../graph_common.hpp"

int main() {
  queue Queue{{sycl::ext::intel::property::queue::no_immediate_command_list{}}};

  using T = int;

  std::vector<T> DataA(Size), DataB(Size), DataC(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  std::vector<T> ReferenceC(DataC);
  for (unsigned n = 0; n < Iterations; n++) {
    for (size_t i = 0; i < Size; i++) {
      ReferenceC[i] += (DataA[i] + DataB[i]) + 1;
    }
  }

  {
    exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};
    buffer<T> BufferA{DataA.data(), range<1>{DataA.size()}};
    BufferA.set_write_back(false);
    buffer<T> BufferB{DataB.data(), range<1>{DataB.size()}};
    BufferB.set_write_back(false);
    buffer<T> BufferC{DataC.data(), range<1>{DataC.size()}};
    BufferC.set_write_back(false);

    Graph.begin_recording(Queue);

    // Create a temporary output buffer to use between kernels.
    {
      buffer<T> BufferTemp{range<1>{DataA.size()}};
      BufferTemp.set_write_back(false);

      // Vector add to temporary output buffer
      Queue.submit([&](handler &CGH) {
        auto PtrA = BufferA.get_access<access::mode::read>(CGH);
        auto PtrB = BufferB.get_access<access::mode::read>(CGH);
        auto PtrOut = BufferTemp.get_access<access::mode::write>(CGH);
        CGH.parallel_for(range<1>(Size),
                         [=](item<1> id) { PtrOut[id] = PtrA[id] + PtrB[id]; });
      });

      // Modify temp buffer and write to output buffer
      Queue.submit([&](handler &CGH) {
        auto PtrTemp = BufferTemp.get_access<access::mode::read>(CGH);
        auto PtrOut = BufferC.get_access<access::mode::write>(CGH);
        CGH.parallel_for(range<1>(Size),
                         [=](item<1> id) { PtrOut[id] += PtrTemp[id] + 1; });
      });
      Graph.end_recording();
    }
    auto GraphExec = Graph.finalize();

    event Event;
    for (unsigned n = 0; n < Iterations; n++) {
      Event = Queue.submit([&](handler &CGH) {
        CGH.depends_on(Event);
        CGH.ext_oneapi_graph(GraphExec);
      });
    }
    Queue.wait_and_throw();

    host_accessor HostAccC(BufferC);
    for (size_t i = 0; i < Size; i++) {
      assert(check_value(i, ReferenceC[i], HostAccC[i], "HostAccC"));
    }
  }

  return 0;
}
