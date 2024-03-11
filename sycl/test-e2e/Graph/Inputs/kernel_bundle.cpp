// Tests using a bundle in a graph.

#include "../graph_common.hpp"

class Kernel1Name;

int main() {
  using T = int;

  const sycl::device Dev{sycl::default_selector_v};
  const sycl::context Ctx{Dev};

  queue Queue{Ctx, Dev};

  sycl::kernel_id KernelID = sycl::get_kernel_id<Kernel1Name>();

  sycl::kernel_bundle KernelBundleInput =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev});
  assert(KernelBundleInput.has_kernel(KernelID));
  assert(sycl::has_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev}));

  sycl::kernel_bundle<sycl::bundle_state::executable> KernelBundleExecutable =
      sycl::build(KernelBundleInput, KernelBundleInput.get_devices());

  sycl::buffer<T, 1> Buf(sycl::range<1>{1});
  Buf.set_write_back(false);
  {
    exp_ext::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

    add_node(Graph, Queue, ([&](sycl::handler &CGH) {
               auto Acc = Buf.get_access<sycl::access::mode::write>(CGH);
               CGH.use_kernel_bundle(KernelBundleExecutable);
               CGH.single_task<Kernel1Name>([=]() { Acc[0] = 42; });
             }));

    auto GraphExec = Graph.finalize();

    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
    Queue.wait_and_throw();
  }
  host_accessor HostAcc(Buf);
  assert(HostAcc[0] == 42);

  return 0;
}
