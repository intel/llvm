// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests using dynamic command groups to update a host task node that also uses
// buffers/accessors
#include "../graph_common.hpp"

using T = int;

void calculate_reference_data(size_t Iterations, size_t Size,
                              std::vector<T> &ReferenceA,
                              std::vector<T> &ReferenceB,
                              std::vector<T> &ReferenceC, T ModValue) {
  for (size_t n = 0; n < Iterations; n++) {
    for (size_t i = 0; i < Size; i++) {
      ReferenceA[i]++;
      ReferenceB[i] += ReferenceA[i];
      ReferenceC[i] -= ReferenceA[i];
      ReferenceB[i]--;
      ReferenceC[i]--;
      ReferenceC[i] += ModValue;
      ReferenceC[i] *= 2;
    }
  }
}

int main() {
  queue Queue{};

  std::vector<T> DataA(Size), DataB(Size), DataC(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  auto DataA2 = DataA;
  auto DataB2 = DataB;
  auto DataC2 = DataC;

  const T ModValue = 7;
  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);
  calculate_reference_data(Iterations, Size, ReferenceA, ReferenceB, ReferenceC,
                           ModValue);
  // We're not updating the buffers here so we need to stack the reference
  // calculations for after the host-task update
  std::vector<T> ReferenceA2(ReferenceA), ReferenceB2(ReferenceB),
      ReferenceC2(ReferenceC);
  calculate_reference_data(Iterations, Size, ReferenceA2, ReferenceB2,
                           ReferenceC2, ModValue * 2);

  buffer<T> BufferA{DataA.data(), range<1>{DataA.size()}};
  BufferA.set_write_back(false);
  buffer<T> BufferB{DataB.data(), range<1>{DataB.size()}};
  BufferB.set_write_back(false);
  buffer<T> BufferC{DataC.data(), range<1>{DataC.size()}};
  BufferC.set_write_back(false);

  {
    exp_ext::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

    // Add some commands to the graph
    auto LastOperation = add_kernels(Graph, Size, BufferA, BufferB, BufferC);

    // Create two different command groups for the host task
    auto CGFA = [&](handler &CGH) {
      auto AccC = BufferC.get_access<access::mode::read_write,
                                     access::target::host_task>(CGH);
      CGH.host_task([=]() {
        for (size_t i = 0; i < Size; i++) {
          AccC[i] += ModValue;
        }
      });
    };
    auto CGFB = [&](handler &CGH) {
      auto AccC = BufferC.get_access<access::mode::read_write,
                                     access::target::host_task>(CGH);
      CGH.host_task([=]() {
        for (size_t i = 0; i < Size; i++) {
          AccC[i] += ModValue * 2;
        }
      });
    };

    auto DynamicCG = exp_ext::dynamic_command_group(Graph, {CGFA, CGFB});
    // Add a host task which modifies PtrC
    auto HostTaskNode = Graph.add(
        DynamicCG, exp_ext::property::node::depends_on{LastOperation});

    // Add another node that depends on the host-task
    Graph.add(
        [&](handler &CGH) {
          auto AccC = BufferC.get_access(CGH);
          CGH.parallel_for(range<1>(Size), [=](item<1> Item) {
            AccC[Item.get_linear_id()] *= 2;
          });
        },
        exp_ext::property::node::depends_on{HostTaskNode});

    auto GraphExec = Graph.finalize(exp_ext::property::graph::updatable{});

    // Execute several Iterations of the graph with the first host-task (CGFA)
    event Event;
    for (unsigned n = 0; n < Iterations; n++) {
      Event = Queue.submit([&](handler &CGH) {
        CGH.depends_on(Event);
        CGH.ext_oneapi_graph(GraphExec);
      });
    }
    Queue.wait_and_throw();
    Queue.copy(BufferA.get_access(), DataA.data());
    Queue.copy(BufferB.get_access(), DataB.data());
    Queue.copy(BufferC.get_access(), DataC.data());
    Queue.wait_and_throw();
    for (size_t i = 0; i < Size; i++) {
      assert(check_value(i, ReferenceA[i], DataA[i], "DataA"));
      assert(check_value(i, ReferenceB[i], DataB[i], "DataB"));
      assert(check_value(i, ReferenceC[i], DataC[i], "DataC"));
    }

    // Update to CGFB
    DynamicCG.set_active_index(1);
    GraphExec.update(HostTaskNode);

    // Execute several Iterations of the graph for second host-task (CGFB)
    for (unsigned n = 0; n < Iterations; n++) {
      Event = Queue.submit([&](handler &CGH) {
        CGH.depends_on(Event);
        CGH.ext_oneapi_graph(GraphExec);
      });
    }

    Queue.wait_and_throw();
    Queue.copy(BufferA.get_access(), DataA2.data());
    Queue.copy(BufferB.get_access(), DataB2.data());
    Queue.copy(BufferC.get_access(), DataC2.data());
    Queue.wait_and_throw();
    for (size_t i = 0; i < Size; i++) {
      assert(check_value(i, ReferenceA2[i], DataA2[i], "DataA2"));
      assert(check_value(i, ReferenceB2[i], DataB2[i], "DataB2"));
      assert(check_value(i, ReferenceC2[i], DataC2[i], "DataC2"));
    }
  }

  return 0;
}
