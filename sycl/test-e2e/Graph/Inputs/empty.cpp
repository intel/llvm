// Tests the empty() method on modifiable command graphs.

#include "../graph_common.hpp"

#include <sycl/properties/all_properties.hpp>

int main() {
  queue Queue{{property::queue::in_order{}}};

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  // Test that empty() returns true when graph has 0 nodes
  assert(Graph.empty() && "Graph should be empty with 0 nodes");

  add_node(Graph, Queue, [&](handler &CGH) { CGH.single_task([=]() {}); });

  // Test that empty() returns false when graph has 1 node
  assert(!Graph.empty() && "Graph should not be empty with 1 node");

  return 0;
}
