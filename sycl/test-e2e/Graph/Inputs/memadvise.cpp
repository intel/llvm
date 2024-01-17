// Tests adding a USM mem advise operation as a graph node.

#include "../graph_common.hpp"

static constexpr int Count = 100;

int main() {

  using T = int;

  queue Queue{{sycl::ext::intel::property::queue::no_immediate_command_list{}}};

  if (!are_graphs_supported(Queue)) {
    return 0;
  }

  if (!Queue.get_device().get_info<info::device::usm_shared_allocations>()) {
    return 0;
  }

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  T *Src = (T *)malloc_shared(sizeof(T) * Count, Queue.get_device(),
                              Queue.get_context());
  T *Dest = (T *)malloc_shared(sizeof(T) * Count, Queue.get_device(),
                               Queue.get_context());
  for (int i = 0; i < Count; i++)
    Src[i] = i;

  {
    // Test handler::mem_advise
    auto Init = add_node(Graph, Queue, [&](handler &CGH) {
      CGH.mem_advise(Src, sizeof(T) * Count, 0);
    });

    add_node(
        Graph, Queue,
        [&](handler &CGH) {
          depends_on_helper(CGH, Init);
          CGH.single_task<class double_dest>([=]() {
            for (int i = 0; i < Count; i++)
              Dest[i] = 2 * Src[i];
          });
        },
        Init);

    auto ExecGraph = Graph.finalize();

    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });
    Queue.wait_and_throw();

    for (int i = 0; i < Count; i++) {
      assert(Dest[i] == i * 2);
    }
  }

  free(Src, Queue);
  free(Dest, Queue);
  return 0;
}
