// RUN: %{build} %{embed-ir} -o %t.out
// RUN: env SYCL_UR_TRACE=2 %{run} %t.out 2>&1 | FileCheck %s

// REQUIRES: linux, fusion, (level_zero || cuda)

// Test complete graph fusion with a memory cmd using buffer

// CHECK-COUNT-1: urCommandBufferAppendKernelLaunchExp
// CHECK-NOT: urCommandBufferAppendKernelLaunchExp

#include "graph_fusion_common.hpp"

int main() {
  int in1[Size], in2[Size], in3[Size], out[Size], ref[Size];
  int zeros[Size];
  int tmp1;
  for (size_t i = 0; i < Size; ++i) {
    zeros[i] = 0;
    in1[i] = i * 2;
    in2[i] = i * 3;
    in3[i] = i * 4;
    out[i] = -1;
    tmp1 = in1[i] + in2[i];
    ref[i] = tmp1 * in3[i];
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
  sycl::buffer<int> bTmp{zeros, sycl::range{Size}};
  bTmp.set_write_back(false);
  sycl::buffer<int> bOut{out, sycl::range{Size}};
  bOut.set_write_back(false);
  {
    sycl_ext::command_graph graph{
        q.get_context(), q.get_device(),
        sycl_ext::property::graph::assume_buffer_outlives_graph{}};

    graph.begin_recording(q);

    q.submit([&](sycl::handler &cgh) {
      auto accIn = bIn.get_access(cgh);
      auto accIn0 = bIn0.get_access(cgh);
      cgh.copy(accIn, accIn0);
    });

    q.submit([&](sycl::handler &cgh) {
      auto accIn0 = bIn0.get_access(cgh);
      auto accIn1 = bIn1.get_access(cgh);
      cgh.copy(accIn0, accIn1);
    });

    graph.end_recording();

    add_nodes(q, graph, bIn1, bIn2, bIn3, bTmp, bOut,
              sycl::property_list{sycl_ext::access_scope_work_item,
                                  sycl_ext::fusion_internal_memory,
                                  sycl::no_init});

    // Trigger fusion during finalization.
    auto exec_graph =
        graph.finalize({sycl_ext::property::graph::require_fusion{}});

    q.ext_oneapi_graph(exec_graph);

    q.wait();
  }

  sycl::host_accessor HostAccOut(bOut);
  sycl::host_accessor HostAccTmp(bTmp);
  // Check the results
  for (size_t i = 0; i < Size; ++i) {
    assert(HostAccOut[i] == (ref[i]) && "Computation error");
    assert((HostAccTmp[i] == zeros[i]) && "Not internalizing");
  }

  return 0;
}
