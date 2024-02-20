// Tests using multiple kernel bundles in a graph.

#include "../graph_common.hpp"

class Kernel1Name;
class Kernel2Name;

int main() {
  using T = int;

  const sycl::device Dev{sycl::default_selector_v};
  const sycl::device Dev2{sycl::default_selector_v};

  const sycl::context Ctx{Dev};
  const sycl::context Ctx2{Dev2};

  queue Queue{Ctx, Dev};

  if (!are_graphs_supported(Queue)) {
    return 0;
  }

  sycl::kernel_id Kernel1ID = sycl::get_kernel_id<Kernel1Name>();
  sycl::kernel_id Kernel2ID = sycl::get_kernel_id<Kernel2Name>();

  sycl::kernel_bundle KernelBundleInput1 =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev},
                                                         {Kernel1ID});
  sycl::kernel_bundle KernelBundleInput2 =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx2, {Dev2},
                                                         {Kernel2ID});
  assert(KernelBundleInput1.has_kernel(Kernel1ID));
  assert(sycl::has_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev}));
  assert(KernelBundleInput2.has_kernel(Kernel2ID));
  assert(sycl::has_kernel_bundle<sycl::bundle_state::input>(Ctx2, {Dev2}));

  sycl::kernel_bundle<sycl::bundle_state::executable> KernelBundleExecutable1 =
      sycl::build(KernelBundleInput1, KernelBundleInput1.get_devices());

  sycl::kernel_bundle<sycl::bundle_state::executable> KernelBundleExecutable2 =
      sycl::build(KernelBundleInput2, KernelBundleInput2.get_devices());

  std::vector<T> DataA(Size);
  std::iota(DataA.begin(), DataA.end(), 1);
  std::vector<T> ReferenceA;
  for (size_t i = 0; i < Size; i++) {
    ReferenceA.push_back(DataA[i] + 1);
  }

  buffer<T> BufferA{DataA.data(), range<1>{DataA.size()}};
  BufferA.set_write_back(false);
  sycl::buffer<T, 1> Buf1(sycl::range<1>{1});
  Buf1.set_write_back(false);
  sycl::buffer<T, 1> Buf2(sycl::range<1>{1});
  Buf2.set_write_back(false);
  {
    exp_ext::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

    add_node(Graph, Queue, ([&](sycl::handler &CGH) {
               auto Acc = Buf1.get_access<sycl::access::mode::write>(CGH);
               CGH.use_kernel_bundle(KernelBundleExecutable1);
               CGH.single_task<Kernel1Name>([=]() { Acc[0] = 42; });
             }));

    add_node(Graph, Queue, ([&](handler &CGH) {
               auto DataA =
                   BufferA.template get_access<access::mode::read_write>(CGH);
               CGH.use_kernel_bundle(KernelBundleExecutable1);
               CGH.parallel_for(range<1>{Size},
                                [=](item<1> Id) { DataA[Id]++; });
             }));

#ifdef GRAPH_E2E_EXPLICIT
    // KernelBundleExecutable2 and the Graph don't share the same context
    // We should therefore get a exception
    // Note we can't do the same test for Record&Replay interface since two
    // queues with different contexts cannot be recorded by the same Graph.
    std::error_code ExceptionCode = make_error_code(sycl::errc::success);
    try {
      Graph.add([&](sycl::handler &CGH) {
        auto Acc = Buf2.get_access<sycl::access::mode::write>(CGH);
        CGH.use_kernel_bundle(KernelBundleExecutable2);
        CGH.single_task<Kernel2Name>([=]() { Acc[0] = 24; });
      });
    } catch (exception &Exception) {
      ExceptionCode = Exception.code();
    }
    assert(ExceptionCode == sycl::errc::invalid);
#else
    // If Explicit API is not used, we still need to add kernel2Name to the
    // bundle since this test expected to find it in the bundle whatever the
    // API used.
    if (0) {
      Queue.submit(
          [](sycl::handler &CGH) { CGH.single_task<Kernel2Name>([]() {}); });
    }
#endif

    auto GraphExec = Graph.finalize();

    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
    Queue.wait_and_throw();
  }
  host_accessor HostAcc1(Buf1);
  assert(HostAcc1[0] == 42);

  host_accessor HostAccA(BufferA);
  for (size_t i = 0; i < Size; i++)
    assert(check_value(i, ReferenceA[i], HostAccA[i], "HostAccA"));

  return 0;
}
