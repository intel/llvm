#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

constexpr size_t Size = 512; // Number of data elements in a buffer.

namespace sycl_ext = sycl::ext::oneapi::experimental;

struct AddKernel {
  sycl::accessor<int, 1> accIn1;
  sycl::accessor<int, 1> accIn2;
  sycl::accessor<int, 1> accOut;

  void operator()(sycl::id<1> i) const { accOut[i] = accIn1[i] + accIn2[i]; }
};

void add_nodes(sycl::queue Queue,
               sycl_ext::command_graph<sycl_ext::graph_state::modifiable> Graph,
               sycl::buffer<int> In1, sycl::buffer<int> In2,
               sycl::buffer<int> In3, sycl::buffer<int> Tmp,
               sycl::buffer<int> Out, sycl::property_list PropAccessors) {
  Graph.begin_recording(Queue);

  Queue.submit([&](sycl::handler &cgh) {
    auto accIn1 = In1.get_access(cgh);
    auto accIn2 = In2.get_access(cgh);
    // Internalization specified on each accessor.
    auto accTmp = Tmp.get_access(cgh, PropAccessors);
    cgh.parallel_for<AddKernel>(Size, AddKernel{accIn1, accIn2, accTmp});
  });

  Queue.submit([&](sycl::handler &cgh) {
    // Internalization specified on each accessor.
    auto accTmp = Tmp.get_access(cgh, PropAccessors);
    auto accIn3 = In3.get_access(cgh);
    auto accOut = Out.get_access(cgh);
    cgh.parallel_for<class KernelOne>(
        Size, [=](sycl::id<1> i) { accOut[i] = accTmp[i] * accIn3[i]; });
  });

  Graph.end_recording();
}
