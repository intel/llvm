// Tests whole graph update with a host-task in the graph using accessors

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

void add_nodes_to_graph(
    exp_ext::command_graph<exp_ext::graph_state::modifiable> &Graph,
    queue &Queue, buffer<T> &BufferA, buffer<T> &BufferB, buffer<T> &BufferC,
    const T &ModValue) {
  // Add some commands to the graph
  auto LastOperation = add_nodes(Graph, Queue, Size, BufferA, BufferB, BufferC);

  // Add a host task which modifies PtrC
  auto HostTaskOp = add_node(
      Graph, Queue,
      [&](handler &CGH) {
        auto AccC = BufferC.get_access<access::mode::read_write,
                                       access::target::host_task>(CGH);
        depends_on_helper(CGH, LastOperation);
        CGH.host_task([=]() {
          for (size_t i = 0; i < Size; i++) {
            AccC[i] += ModValue;
          }
        });
      },
      LastOperation);

  // Add another node that depends on the host-task
  add_node(
      Graph, Queue,
      [&](handler &CGH) {
        auto AccC = BufferC.get_access(CGH);
        depends_on_helper(CGH, HostTaskOp);
        CGH.parallel_for(range<1>(Size), [=](item<1> Item) {
          AccC[Item.get_linear_id()] *= 2;
        });
      },
      HostTaskOp);
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

  buffer<T> BufferA{DataA.data(), range<1>{DataA.size()}};
  buffer<T> BufferB{DataB.data(), range<1>{DataB.size()}};
  buffer<T> BufferC{DataC.data(), range<1>{DataC.size()}};
  buffer<T> BufferA2{DataA2.data(), range<1>{DataA2.size()}};
  buffer<T> BufferB2{DataB2.data(), range<1>{DataB2.size()}};
  buffer<T> BufferC2{DataC2.data(), range<1>{DataC2.size()}};
  BufferA.set_write_back(false);
  BufferB.set_write_back(false);
  BufferC.set_write_back(false);
  BufferA2.set_write_back(false);
  BufferB2.set_write_back(false);
  BufferC2.set_write_back(false);

  {
    exp_ext::command_graph GraphA{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

    // Fill graphA with nodes
    add_nodes_to_graph(GraphA, Queue, BufferA, BufferB, BufferC, ModValue);

    auto GraphExec = GraphA.finalize(exp_ext::property::graph::updatable{});

    exp_ext::command_graph GraphB{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

    // Fill graphB with nodes, with a different set of pointers
    add_nodes_to_graph(GraphB, Queue, BufferA2, BufferB2, BufferC2, ModValue);

    // Execute several Iterations of the graph for 1st set of buffers
    event Event;
    for (unsigned n = 0; n < Iterations; n++) {
      Event = Queue.submit([&](handler &CGH) {
        CGH.depends_on(Event);
        CGH.ext_oneapi_graph(GraphExec);
      });
    }
    Queue.wait_and_throw();

    GraphExec.update(GraphB);

    // Execute several Iterations of the graph for 2nd set of buffers
    for (unsigned n = 0; n < Iterations; n++) {
      Event = Queue.submit([&](handler &CGH) {
        CGH.depends_on(Event);
        CGH.ext_oneapi_graph(GraphExec);
      });
    }

    Queue.wait_and_throw();
  }
  Queue.copy(BufferA.get_access(), DataA.data());
  Queue.copy(BufferB.get_access(), DataB.data());
  Queue.copy(BufferC.get_access(), DataC.data());
  Queue.copy(BufferA2.get_access(), DataA2.data());
  Queue.copy(BufferB2.get_access(), DataB2.data());
  Queue.copy(BufferC2.get_access(), DataC2.data());
  Queue.wait_and_throw();

  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, ReferenceA[i], DataA[i], "DataA"));
    assert(check_value(i, ReferenceB[i], DataB[i], "DataB"));
    assert(check_value(i, ReferenceC[i], DataC[i], "DataC"));

    assert(check_value(i, ReferenceA[i], DataA2[i], "DataA2"));
    assert(check_value(i, ReferenceB[i], DataB2[i], "DataB2"));
    assert(check_value(i, ReferenceC[i], DataC2[i], "DataC2"));
  }

  return 0;
}
