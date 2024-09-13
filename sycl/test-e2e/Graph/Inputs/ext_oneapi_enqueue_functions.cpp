// Tests the enqueue free function kernel shortcuts.

#include "../graph_common.hpp"
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/properties/all_properties.hpp>

template <typename T>
void run_kernels_usm_in_order(queue Q, const size_t Size, T *DataA, T *DataB,
                              T *DataC, std::vector<T> &Output, T Pattern) {
  exp_ext::fill(Q, DataA, Pattern, Size);

  exp_ext::single_task(Q, [=]() {
    for (size_t i = 0; i < Size; ++i) {
      DataB[i] = i;
    }
  });

  exp_ext::parallel_for(Q, sycl::range<1>{Size}, [=](sycl::item<1> Item) {
    DataC[Item] = DataA[Item] * DataB[Item];
  });

  exp_ext::copy(Q, DataC, Output.data(), Size);
}

template <typename T>
void add_kernels_usm_in_order(
    exp_ext::command_graph<exp_ext::graph_state::modifiable> G,
    const size_t Size, T *DataA, T *DataB, T *DataC, std::vector<T> &Output,
    T Pattern) {
  exp_ext::submit(G,
                  [&](sycl::handler &CGH) { CGH.fill(DataA, Pattern, Size); });

  exp_ext::submit(G, [&](sycl::handler &CGH) {
    CGH.single_task([=]() {
      for (size_t i = 0; i < Size; ++i) {
        DataB[i] = i;
      }
    });
  });

  exp_ext::submit(G, [&](sycl::handler &CGH) {
    CGH.parallel_for(sycl::range<1>{Size}, [=](sycl::item<1> Item) {
      DataC[Item] = DataA[Item] * DataB[Item];
    });
  });

  exp_ext::submit(
      G, [&](sycl::handler &CGH) { CGH.copy(DataC, Output.data(), Size); });
}

template <typename T>
void add_nodes_in_order(
    exp_ext::command_graph<exp_ext::graph_state::modifiable> Graph, queue Queue,
    const size_t Size, T *DataA, T *DataB, T *DataC, std::vector<T> &Output,
    T Pattern) {
#if defined(GRAPH_E2E_EXPLICIT)
  add_kernels_usm_in_order(Graph, Size, DataA, DataB, DataC, Output, Pattern);
#elif defined(GRAPH_E2E_RECORD_REPLAY)
  Graph.begin_recording(Queue);
  run_kernels_usm_in_order(Queue, Size, DataA, DataB, DataC, Output, Pattern);
  Graph.end_recording(Queue);
#else
  assert(0 && "Error: Cannot use add_nodes without selecting an API");
#endif
}

int main() {
  queue InOrderQueue{property::queue::in_order{}};

  using T = int;
  T Pattern = 42;

  T *PtrA = malloc_device<T>(Size, InOrderQueue);
  T *PtrB = malloc_device<T>(Size, InOrderQueue);
  T *PtrC = malloc_device<T>(Size, InOrderQueue);

  std::vector<T> Output(Size);

  exp_ext::command_graph Graph{InOrderQueue};

  add_nodes_in_order(Graph, InOrderQueue, Size, PtrA, PtrB, PtrC, Output,
                     Pattern);

  auto GraphExec = Graph.finalize();

  InOrderQueue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
  InOrderQueue.wait_and_throw();

  free(PtrA, InOrderQueue);
  free(PtrB, InOrderQueue);
  free(PtrC, InOrderQueue);

  for (size_t i = 0; i < Size; i++) {
    T Ref = Pattern * i;
    assert(Output[i] == Ref);
  }

  return 0;
}
