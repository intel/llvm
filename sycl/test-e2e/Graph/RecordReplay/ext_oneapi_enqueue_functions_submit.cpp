// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests the enqueue free function using buffers and submit

#include "../graph_common.hpp"
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>

int main() {
  queue Queue{};

  using T = int;

  buffer<T> BufferA{range<1>{Size}};
  BufferA.set_write_back(false);
  buffer<T> BufferB{range<1>{Size}};
  BufferB.set_write_back(false);
  buffer<T> BufferC{range<1>{Size}};
  BufferC.set_write_back(false);

  const T Pattern = 42;
  {

    exp_ext::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

    Graph.begin_recording(Queue);

    exp_ext::submit(Queue, [&](handler &CGH) {
      accessor AccA{BufferA, CGH, write_only};
      exp_ext::single_task(CGH, [=]() {
        for (size_t i = 0; i < Size; i++) {
          AccA[i] = Pattern;
        }
      });
    });

    exp_ext::submit(Queue, [&](handler &CGH) {
      accessor AccB{BufferB, CGH, write_only};
      exp_ext::parallel_for(CGH, range<1>{Size},
                            [=](item<1> Item) { AccB[Item] = Item; });
    });

    exp_ext::submit(Queue, [&](handler &CGH) {
      accessor AccA{BufferA, CGH, read_only};
      accessor AccB{BufferB, CGH, read_only};
      accessor AccC{BufferC, CGH, write_only};
      exp_ext::parallel_for(CGH, range<1>{Size}, [=](item<1> Item) {
        AccC[Item] = AccA[Item] * AccB[Item];
      });
    });

    Graph.end_recording();

    auto GraphExec = Graph.finalize();

    exp_ext::execute_graph(Queue, GraphExec);
    Queue.wait_and_throw();
  }

  host_accessor HostAcc(BufferC);

  for (size_t i = 0; i < Size; i++) {
    T Ref = Pattern * i;
    assert(HostAcc[i] == Ref);
  }

  return 0;
}
