// RUN: %{build} %{embed-ir} -o %t.out
// RUN: env SYCL_UR_TRACE=2 %{run} %t.out 2>&1 | FileCheck %s

// REQUIRES: linux, fusion, (level_zero || cuda)

// Test a fused graph added as a subgraph.

// This tests creates 1 exec_graph made of 1 fused kernel.
// Then the main graph is composed of the fused graph (as part of the subgraph)
// memory cmd before the fused graph and another single kernel.
// So we must find only 3 kernel enqueing calls
// CHECK-COUNT-3: urCommandBufferAppendKernelLaunchExp
// CHECK-NOT: urCommandBufferAppendKernelLaunchExp

#include "graph_fusion_common.hpp"

int main() {
  int in1[Size], in2[Size], in3[Size], out[Size], ref[Size], zeros[Size];
  int tmp1;
  for (size_t i = 0; i < Size; ++i) {
    zeros[i] = 0;
    in1[i] = i * 2;
    in2[i] = i * 3;
    in3[i] = i * 4;
    out[i] = -1;
    tmp1 = in1[i] + in2[i];
    ref[i] = (tmp1 * in3[i]) * 2;
  }

  sycl::queue q{sycl::property::queue::in_order{}};
  sycl::buffer<int> bIn{in1, sycl::range{Size}};
  bIn.set_write_back(false);
  sycl::buffer<int> bIn0{zeros, sycl::range{Size}};
  bIn0.set_write_back(false);
  sycl::buffer<int> bIn1{zeros, sycl::range{Size}};
  bIn1.set_write_back(false);
  sycl::buffer<int> bIn2{in2, sycl::range{Size}};
  bIn2.set_write_back(false);
  sycl::buffer<int> bIn3{in3, sycl::range{Size}};
  bIn3.set_write_back(false);
  sycl::buffer<int> bTmp{zeros,
                         sycl::range{Size},
                         {sycl_ext::access_scope_work_item,
                          sycl_ext::fusion_internal_memory, sycl::no_init}};
  bTmp.set_write_back(false);
  sycl::buffer<int> bOut{out, sycl::range{Size}};
  bOut.set_write_back(false);
  sycl::buffer<int> bOutFinal{out, sycl::range{Size}};
  bOutFinal.set_write_back(false);
  {
    sycl_ext::command_graph graph{
        q.get_context(), q.get_device(),
        sycl_ext::property::graph::assume_buffer_outlives_graph{}};

    add_nodes(q, graph, bIn1, bIn2, bIn3, bTmp, bOut, {});

    graph.end_recording();

    auto exec_graph =
        graph.finalize({sycl_ext::property::graph::require_fusion{}});

    // Add node to main graph followed by sub-graph and another node
    sycl_ext::command_graph main_graph{
        q.get_context(), q.get_device(),
        sycl_ext::property::graph::assume_buffer_outlives_graph{}};

    main_graph.begin_recording(q);

    q.submit([&](sycl::handler &cgh) {
      sycl::accessor accIn(bIn, cgh);
      sycl::accessor accIn0(bIn0, cgh);
      cgh.copy(accIn, accIn0);
    });

    q.submit([&](sycl::handler &cgh) {
      sycl::accessor accIn0(bIn0, cgh);
      sycl::accessor accIn1(bIn1, cgh);
      cgh.copy(accIn0, accIn1);
    });

    main_graph.end_recording();

    main_graph.add(
        [&](sycl::handler &CGH) { CGH.ext_oneapi_graph(exec_graph); });

    main_graph.begin_recording(q);

    q.submit([&](sycl::handler &cgh) {
      sycl::accessor accOut(bOut, cgh);
      sycl::accessor accOutFinal(bOutFinal, cgh);
      cgh.parallel_for<class KernelDouble>(
          Size, [=](sycl::id<1> i) { accOutFinal[i] = accOut[i] * 2; });
    });

    main_graph.end_recording();

    auto exec_main_graph = main_graph.finalize();

    q.ext_oneapi_graph(exec_main_graph);

    q.wait();
  }

  sycl::host_accessor HostAccOut(bOutFinal);
  sycl::host_accessor HostAccTmp1(bTmp);
  // Check the results
  for (size_t i = 0; i < Size; ++i) {
    assert((HostAccOut[i] == ref[i]) && "Computation error");
    assert((HostAccTmp1[i] == zeros[i]) && "Not internalizing");
  }

  return 0;
}
