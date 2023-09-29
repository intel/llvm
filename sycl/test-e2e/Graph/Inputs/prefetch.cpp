// Tests adding a USM Prefetch operation as a graph node.

#include "../graph_common.hpp"

static constexpr int Count = 100;

int main() {

  using T = int;

  queue Queue{{sycl::ext::intel::property::queue::no_immediate_command_list{}}};

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

  // Test handler::prefetch
  {
    auto InitPrefetch = add_node(Graph, Queue, [&](handler &CGH) {
      CGH.prefetch(Src, sizeof(T) * Count);
    });

    add_node(
        Graph, Queue,
        [&](handler &CGH) {
          CGH.single_task<class double_dest>([=]() {
            for (int i = 0; i < Count; i++)
              Dest[i] = 2 * Src[i];
          });
        },
        InitPrefetch);

    auto ExecGraph = Graph.finalize();

    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });
    Queue.wait_and_throw();

    for (int i = 0; i < Count; i++) {
      assert(Dest[i] == i * 2);
    }
  }

#ifdef GRAPH_E2E_RECORD_REPLAY
  exp_ext::command_graph Graph2{Queue.get_context(), Queue.get_device()};

  // Test queue::prefetch
  {
    Graph2.begin_recording(Queue);

    event InitPrefetch = Queue.prefetch(Src, sizeof(T) * Count);

    Queue.submit([&](handler &CGH) {
      CGH.depends_on(InitPrefetch);
      CGH.single_task<class double_dest3>([=]() {
        for (int i = 0; i < Count; i++)
          Dest[i] = 3 * Src[i];
      });
    });
    Graph2.end_recording();

    auto ExecGraph2 = Graph2.finalize();

    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph2); });
    Queue.wait_and_throw();

    for (int i = 0; i < Count; i++) {
      assert(Dest[i] == i * 3);
    }
  }
#endif

  free(Src, Queue);
  free(Dest, Queue);
  return 0;
}
