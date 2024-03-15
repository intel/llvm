// This test checks the querying of the state of an event
// returned from graph submission
// with event::get_info<info::event::command_execution_status>()
// An event should pass from the submitted state to the complete state.
// The running state seems to not be implemented by the level_zero backend.
// This test should display (in most execution environment):
// -----
// submitted
// complete
// -----
// However, the execution support may be fast enough to complete
// the computation before we reach the state monitoring query.
// In this case, the displayed output can be:
// -----
// complete
// complete
// -----
// We therefore only check that the complete state of the event
// in this test.

#include "../graph_common.hpp"

std::string event_status_name(sycl::info::event_command_status status) {
  switch (status) {
  case sycl::info::event_command_status::submitted:
    return "submitted";
  case sycl::info::event_command_status::running:
    return "running";
  case sycl::info::event_command_status::complete:
    return "complete";
  default:
    return "unknown (" + std::to_string(int(status)) + ")";
  }
}

int main() {
  queue Queue{};

  using T = int;

  const T ModValue = 7;
  std::vector<T> DataA(Size), DataB(Size), DataC(Size);

  std::iota(DataA.begin(), DataA.end(), 1);
  std::iota(DataB.begin(), DataB.end(), 10);
  std::iota(DataC.begin(), DataC.end(), 1000);

  // Create reference data for output
  std::vector<T> ReferenceA(DataA), ReferenceB(DataB), ReferenceC(DataC);
  for (size_t j = 0; j < Size; j++) {
    ReferenceA[j] = ReferenceB[j];
    ReferenceA[j] += ModValue;
    ReferenceB[j] = ReferenceA[j];
    ReferenceB[j] += ModValue;
    ReferenceC[j] = ReferenceB[j];
  }

  buffer BufferA{DataA};
  BufferA.set_write_back(false);
  buffer BufferB{DataB};
  BufferB.set_write_back(false);
  buffer BufferC{DataC};
  BufferC.set_write_back(false);

  {
    exp_ext::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

    // Copy from B to A
    auto Init = add_node(Graph, Queue, [&](handler &CGH) {
      auto AccA = BufferA.get_access(CGH);
      auto AccB = BufferB.get_access(CGH);
      CGH.copy(AccB, AccA);
    });

    // Read & write A
    auto Node1 = add_node(Graph, Queue, [&](handler &CGH) {
      auto AccA = BufferA.get_access(CGH);
      CGH.parallel_for(range<1>(Size), [=](item<1> id) {
        auto LinID = id.get_linear_id();
        AccA[LinID] += ModValue;
      });
    });

    // Read & write B
    auto Node2 = add_node(Graph, Queue, [&](handler &CGH) {
      auto AccB = BufferB.get_access(CGH);
      CGH.parallel_for(range<1>(Size), [=](item<1> id) {
        auto LinID = id.get_linear_id();
        AccB[LinID] += ModValue;
      });
    });

    // memcpy from A to B
    auto Node3 = add_node(Graph, Queue, [&](handler &CGH) {
      auto AccA = BufferA.get_access(CGH);
      auto AccB = BufferB.get_access(CGH);
      CGH.copy(AccA, AccB);
    });

    // Read and write B
    auto Node4 = add_node(Graph, Queue, [&](handler &CGH) {
      auto AccB = BufferB.get_access(CGH);
      CGH.parallel_for(range<1>(Size), [=](item<1> id) {
        auto LinID = id.get_linear_id();
        AccB[LinID] += ModValue;
      });
    });

    // Copy from B to C
    auto Node5 = add_node(Graph, Queue, [&](handler &CGH) {
      auto AccB = BufferB.get_access(CGH);
      auto AccC = BufferC.get_access(CGH);
      CGH.copy(AccB, AccC);
    });

    auto GraphExec = Graph.finalize();

    sycl::event Event =
        Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
    auto Info = Event.get_info<info::event::command_execution_status>();
    std::cout << event_status_name(Info) << std::endl;
    while (
        (Info =
             Event.get_info<sycl::info::event::command_execution_status>()) !=
        sycl::info::event_command_status::complete) {
    }
    std::cout << event_status_name(Info) << std::endl;

    Queue.wait_and_throw();
  }

  host_accessor HostAccA(BufferA);
  host_accessor HostAccB(BufferB);
  host_accessor HostAccC(BufferC);

  for (size_t i = 0; i < Size; i++) {
    assert(check_value(i, ReferenceA[i], HostAccA[i], "HostAccA"));
    assert(check_value(i, ReferenceB[i], HostAccB[i], "HostAccB"));
    assert(check_value(i, ReferenceC[i], HostAccC[i], "HostAccC"));
  }

  return 0;
}
