// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// REQUIRES: aspect-usm_shared_allocations

// Modified example from issue #19450 which identified an issue with updating
// multiple kernel nodes which share the same kernel.

#include "../graph_common.hpp"

int main() {
  sycl::queue q;

  static constexpr size_t R = 10;
  static constexpr size_t I = 5;
  int *output = sycl::malloc_shared<int>(I, q);
  std::fill(output, output + I, 0);

  std::unique_ptr<sycl::ext::oneapi::experimental::command_graph<
      sycl::ext::oneapi::experimental::graph_state::executable>>
      graph;
  for (int r = 0; r < R; ++r) {

    sycl::ext::oneapi::experimental::command_graph<
        sycl::ext::oneapi::experimental::graph_state::modifiable>
        modifiable_graph(q.get_context(), q.get_device());
    for (size_t i = 1; i < I; i++) {
      sycl::range global = {i, i, i};
      sycl::range local = {i, i, i};
      modifiable_graph.add([=](sycl::handler &h) {
        h.parallel_for<class test>(sycl::nd_range{global, local},
                                   [=](sycl::nd_item<3> it) noexcept {
                                     if (it.get_group().leader()) {
                                       output[i]++;
                                     }
                                   });
      });
    }

    if (r == 0) {
      const auto instance = modifiable_graph.finalize(
          sycl::ext::oneapi::experimental::property::graph::updatable{});
      graph = std::make_unique<sycl::ext::oneapi::experimental::command_graph<
          sycl::ext::oneapi::experimental::graph_state::executable>>(
          std::move(instance));
    } else {
      graph->update(modifiable_graph);
    }
    q.ext_oneapi_graph(*graph).wait();
  }

  q.wait();
  std::array<int, I> Ref{0, R, R, R, R};

  for (int i = 0; i < I; ++i) {
    assert(output[i] == Ref[i]);
  }
}
