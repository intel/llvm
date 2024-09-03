// RUN: %{build} %{embed-ir} -o %t.out
// RUN: %{run} %t.out

// REQUIRES: linux, fusion, (level_zero || cuda)

// Test complete fusion with a combination of kernels that require a work-group
// barrier to be inserted by fusion.

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
    ref[i] = tmp1 * in3[i];
  }

  sycl::queue q{sycl::property::queue::in_order{}};
  sycl::buffer<int> bIn1{in1, sycl::range{Size}};
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

    add_nodes(q, graph, bIn1, bIn2, bIn3, bTmp, bOut, {});

    // Trigger fusion during finalization.
    auto exec_graph =
        graph.finalize({sycl_ext::property::graph::require_fusion{},
                        sycl_ext::property::graph::insert_barriers{}});

    q.ext_oneapi_graph(exec_graph);

    q.wait();
  }

  // Check the results
  sycl::host_accessor HostAccOut(bOut);
  for (size_t i = 0; i < Size; ++i) {
    assert((HostAccOut[i] == ref[i]) && "Computation error");
  }

  return 0;
}
