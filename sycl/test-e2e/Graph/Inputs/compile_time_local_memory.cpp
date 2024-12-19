// Tests adding kernel nodes with local memory that is allocated using
// the sycl_ext_oneapi_local_memory extension.

#include "../graph_common.hpp"
#include <sycl/ext/oneapi/group_local_memory.hpp>

int main() {
  queue Queue{};

  using T = int;
  constexpr size_t LocalSize = 128;

  std::vector<T> HostData(Size);
  std::iota(HostData.begin(), HostData.end(), 10);

  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  T *PtrA = malloc_device<T>(Size, Queue);

  Queue.copy(HostData.data(), PtrA, Size);
  Queue.wait_and_throw();

  auto NodeA = add_node(Graph, Queue, [&](handler &CGH) {
    CGH.parallel_for(nd_range({Size}, {LocalSize}), [=](nd_item<1> Item) {
      multi_ptr<size_t[LocalSize], access::address_space::local_space>
          LocalMem = sycl::ext::oneapi::group_local_memory<size_t[LocalSize]>(
              Item.get_group());
      *LocalMem[Item.get_local_linear_id()] = Item.get_global_linear_id() * 2;
      PtrA[Item.get_global_linear_id()] +=
          *LocalMem[Item.get_local_linear_id()];
    });
  });

  add_node(
      Graph, Queue,
      [&](handler &CGH) {
        depends_on_helper(CGH, NodeA);
        CGH.parallel_for(nd_range({Size}, {LocalSize}), [=](nd_item<1> Item) {
          multi_ptr<size_t[LocalSize], access::address_space::local_space>
              LocalMem = sycl::ext::oneapi::group_local_memory_for_overwrite<
                  size_t[LocalSize]>(Item.get_group());
          *LocalMem[Item.get_local_linear_id()] =
              Item.get_global_linear_id() + 4;
          PtrA[Item.get_global_linear_id()] *=
              *LocalMem[Item.get_local_linear_id()];
        });
      },
      NodeA);

  auto GraphExec = Graph.finalize();

  for (unsigned n = 0; n < Iterations; n++) {
    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
  }

  Queue.wait_and_throw();

  Queue.copy(PtrA, HostData.data(), Size);
  Queue.wait_and_throw();

  free(PtrA, Queue);

  for (size_t i = 0; i < Size; i++) {
    T Ref = 10 + i;
    for (size_t iter = 0; iter < Iterations; ++iter) {
      Ref += (i * 2);
      Ref *= (i + 4);
    }
    assert(check_value(i, Ref, HostData[i], "PtrA"));
  }

  return 0;
}
