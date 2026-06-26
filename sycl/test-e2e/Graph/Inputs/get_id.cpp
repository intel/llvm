// Tests for command_graph::get_id()

#include "../graph_common.hpp"
#include <sycl/properties/all_properties.hpp>

int main() {
  device Dev;
  context Ctx{Dev};
  queue Q{Ctx, Dev, {property::queue::in_order{}}};

#ifdef GRAPH_E2E_NATIVE_RECORDING
  const property_list GraphProps{
      exp_ext::property::graph::enable_native_recording{}};
#else
  const property_list GraphProps{};
#endif

  // Test 1: Monotonic uniqueness
  {
    constexpr int N = 5;
    std::vector<exp_ext::command_graph<>> Graphs;
    Graphs.reserve(N);
    for (int i = 0; i < N; ++i)
      Graphs.emplace_back(Ctx, Dev, GraphProps);
    for (int i = 1; i < N; ++i) {
      assert(Graphs[i].get_id() > Graphs[i - 1].get_id() &&
             "IDs must be strictly increasing");
    }

    // All distinct (implied by strictly increasing, but check explicitly).
    for (int i = 0; i < N; ++i)
      for (int j = i + 1; j < N; ++j)
        assert(Graphs[i].get_id() != Graphs[j].get_id() &&
               "IDs must be distinct");
  }

  // Test 2: Stability across begin_recording/end_recording cycles.
  {
    exp_ext::command_graph Graph{Ctx, Dev, GraphProps};
    size_t ID = Graph.get_id();

    Graph.begin_recording(Q);
    assert(Graph.get_id() == ID && "ID must be stable during recording");
    Graph.end_recording(Q);

    assert(Graph.get_id() == ID && "ID must be stable after end_recording");

    // Second cycle.
    Graph.begin_recording(Q);
    assert(Graph.get_id() == ID &&
           "ID must be stable on second recording cycle");
    Graph.end_recording(Q);
  }

  // Test 3: Distinct graphs have distinct IDs.
  {
    exp_ext::command_graph GraphA{Ctx, Dev, GraphProps};
    exp_ext::command_graph GraphB{Ctx, Dev, GraphProps};
    assert(GraphA.get_id() != GraphB.get_id() &&
           "Distinct graphs must have distinct IDs");
  }

  // Test 4: On fork/join, both queues see the same ID.
  {
    queue Q2{Ctx, Dev, {property::queue::in_order{}}};
    exp_ext::command_graph Graph{Ctx, Dev, GraphProps};

    Graph.begin_recording(Q);
    auto ForkEvent = Q.ext_oneapi_submit_barrier();
    // Q2 transitions to recording via transitive dependency.
    auto Q2Event = Q2.ext_oneapi_submit_barrier({ForkEvent});

    Q.ext_oneapi_submit_barrier({Q2Event});

    auto GFromQ1 = Q.ext_oneapi_get_graph();
    auto GFromQ2 = Q2.ext_oneapi_get_graph();

    assert(GFromQ1.get_id() == Graph.get_id() &&
           "Graph ID from Q1 must match original graph ID");
    assert(GFromQ2.get_id() == Graph.get_id() &&
           "Graph ID from Q2 must match original graph ID");
    assert(GFromQ1.get_id() == GFromQ2.get_id() &&
           "Both queues must see the same ID for a fork/join graph");

    Graph.end_recording();
  }

  // Test 5: IDs are unique across contexts.
  {
    context Ctx2{Dev};
    exp_ext::command_graph GraphA{Ctx, Dev, GraphProps};
    exp_ext::command_graph GraphB{Ctx2, Dev, GraphProps};
    assert(GraphA.get_id() != GraphB.get_id() &&
           "Graphs in different contexts must have distinct IDs");
  }

  return 0;
}
