// RUN: %{build} %{embed-ir} -o %t.out
// RUN: env SYCL_UR_TRACE=2 %{run} %t.out 2>&1 | FileCheck %s

// REQUIRES: linux, fusion, (level_zero || cuda)

// Test complete graph fusion using buffer

// This tests runs the fused graph 6 times, so we must find 6 kernel enqueing
// calls
// CHECK-COUNT-6: urCommandBufferAppendKernelLaunchExp
// CHECK-NOT: urCommandBufferAppendKernelLaunchExp

#include "graph_fusion_common.hpp"

void run_fused_graph(int *in1, int *in2, int *in3, int *out, int *ref,
                     int *zeros, sycl::property_list BufferProps,
                     sycl::property_list AccessorProps) {
  sycl::queue q{sycl::property::queue::in_order{}};
  sycl::buffer<int> bIn1{in1, sycl::range{Size}};
  bIn1.set_write_back(false);
  sycl::buffer<int> bIn2{in2, sycl::range{Size}};
  bIn2.set_write_back(false);
  sycl::buffer<int> bIn3{in3, sycl::range{Size}};
  bIn3.set_write_back(false);
  sycl::buffer<int> bTmp{zeros, sycl::range{Size}, BufferProps};
  bTmp.set_write_back(false);
  sycl::buffer<int> bOut{out, sycl::range{Size}};
  bOut.set_write_back(false);
  {
    sycl_ext::command_graph graph{
        q.get_context(), q.get_device(),
        sycl_ext::property::graph::assume_buffer_outlives_graph{}};

    add_nodes(q, graph, bIn1, bIn2, bIn3, bTmp, bOut, AccessorProps);

    // Trigger fusion during finalization.
    auto exec_graph =
        graph.finalize({sycl_ext::property::graph::require_fusion{}});

    q.ext_oneapi_graph(exec_graph);

    q.wait();
  }

  sycl::host_accessor HostAccOut(bOut);
  sycl::host_accessor HostAccTmp1(bTmp);
  // Check the results
  for (size_t i = 0; i < Size; ++i) {
    assert((HostAccOut[i] == ref[i]) && "Computation error");
    assert((HostAccTmp1[i] == zeros[i]) && "Not internalizing");
  }
}

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
    ref[i] = tmp1 * in3[i];
  }

  // Test fusion with Private internalization.
  sycl::property_list PrivateProps = {sycl_ext::access_scope_work_item,
                                      sycl_ext::fusion_internal_memory,
                                      sycl::no_init};
  // Test fusion when properties are passed through accessors
  run_fused_graph(in1, in2, in3, out, ref, zeros, {}, PrivateProps);

  // Test fusion when properties are passed through buffers
  run_fused_graph(in1, in2, in3, out, ref, zeros, PrivateProps, {});

  // Test fusion when properties are passed through buffers and accessors
  run_fused_graph(in1, in2, in3, out, ref, zeros, PrivateProps, PrivateProps);

  // Test fusion with Local internalization.
  sycl::property_list LocalProps = {sycl_ext::access_scope_work_group,
                                    sycl_ext::fusion_internal_memory,
                                    sycl::no_init};
  // Test fusion when properties are passed through accessors
  run_fused_graph(in1, in2, in3, out, ref, zeros, {}, LocalProps);

  // Test fusion when properties are passed through buffers
  run_fused_graph(in1, in2, in3, out, ref, zeros, LocalProps, {});

  // Test fusion when properties are passed through buffers and accessors
  run_fused_graph(in1, in2, in3, out, ref, zeros, LocalProps, LocalProps);

  return 0;
}
