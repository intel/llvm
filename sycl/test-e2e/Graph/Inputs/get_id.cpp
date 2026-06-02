// Tests for command_graph::get_id()

#include "../graph_common.hpp"
#include <sycl/properties/all_properties.hpp>

int main() {
  device Dev;
  context Ctx{Dev};
  queue Q{Ctx, Dev, {property::queue::in_order{}}};

  // Test 1: Monotonic uniqueness
  {
    constexpr int N = 5;
    std::vector<exp_ext::command_graph<>> Graphs;
    Graphs.reserve(N);
    for (int i = 0; i < N; ++i)
      Graphs.emplace_back(Ctx, Dev);
    for (int i = 1; i < N; ++i) {
      assert(Graphs[i].get_id() >
                 Graphs[i - 1].get_id() &&
             "IDs must be strictly increasing");
    }

    // All distinct (implied by strictly increasing, but check explicitly).
    for (int i = 0; i < N; ++i)
      for (int j = i + 1; j < N; ++j)
        assert(Graphs[i].get_id() !=
                   Graphs[j].get_id() &&
               "IDs must be distinct");
  }

  // Test 2: Stability across begin_recording/end_recording cycles.
  {
    exp_ext::command_graph Graph{Ctx, Dev};
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
    exp_ext::command_graph GraphA{Ctx, Dev};
    exp_ext::command_graph GraphB{Ctx, Dev};
    assert(GraphA.get_id() != GraphB.get_id() &&
           "Distinct graphs must have distinct IDs");
  }

  // Test 4: On fork/join, both queues see the same ID.
  {
    queue Q2{Ctx, Dev, {property::queue::in_order{}}};
    exp_ext::command_graph Graph{Ctx, Dev};

    Graph.begin_recording(Q);
    auto ForkEvent = Q.ext_oneapi_submit_barrier();
    // Q2 transitions to recording via transitive dependency.
    Q2.ext_oneapi_submit_barrier({ForkEvent});

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

  return 0;
}
