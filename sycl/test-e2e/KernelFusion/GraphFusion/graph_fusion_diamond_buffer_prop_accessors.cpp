// RUN: %{build} %{embed-ir} -o %t.out
// RUN: env SYCL_UR_TRACE=2 %{run} %t.out 2>&1 | FileCheck %s

// REQUIRES: linux, fusion, (level_zero || cuda)

// Test complete diamond graph fusion using buffer

// CHECK-COUNT-1: urCommandBufferAppendKernelLaunchExp
// CHECK-NOT: urCommandBufferAppendKernelLaunchExp

#include "graph_fusion_common.hpp"

int main() {
  int in1[Size], in2[Size], in3[Size], out[Size], ref[Size], zeros[Size];
  int tmp1, tmp2, tmp3;
  for (size_t i = 0; i < Size; ++i) {
    zeros[i] = 0;
    in1[i] = i * 2;
    in2[i] = i * 3;
    in3[i] = i * 4;
    out[i] = -1;
    tmp1 = in1[i] + in2[i];
    tmp2 = tmp1 * in3[i];
    tmp3 = tmp1 * 5;
    ref[i] = tmp2 + tmp3;
  }

  sycl::queue q{};
  sycl::buffer<int> bIn1{in1, sycl::range{Size}};
  bIn1.set_write_back(false);
  sycl::buffer<int> bIn2{in2, sycl::range{Size}};
  bIn2.set_write_back(false);
  sycl::buffer<int> bIn3{in3, sycl::range{Size}};
  bIn3.set_write_back(false);
  sycl::buffer<int> bTmp1{zeros, sycl::range{Size}};
  bTmp1.set_write_back(false);
  sycl::buffer<int> bTmp2{zeros, sycl::range{Size}};
  bTmp2.set_write_back(false);
  sycl::buffer<int> bTmp3{zeros, sycl::range{Size}};
  bTmp3.set_write_back(false);
  sycl::buffer<int> bOut{out, sycl::range{Size}};
  bOut.set_write_back(false);
  {
    sycl_ext::command_graph graph{
        q.get_context(), q.get_device(),
        sycl_ext::property::graph::assume_buffer_outlives_graph{}};

    graph.begin_recording(q);

    auto Node1 = q.submit([&](sycl::handler &cgh) {
      auto accIn1 = bIn1.get_access(cgh);
      auto accIn2 = bIn2.get_access(cgh);
      // Internalization specified on each accessor.
      auto accTmp1 = bTmp1.get_access(
          cgh,
          sycl::property_list{sycl_ext::access_scope_work_item,
                              sycl_ext::fusion_internal_memory, sycl::no_init});
      cgh.parallel_for<AddKernel>(Size, AddKernel{accIn1, accIn2, accTmp1});
    });

    auto Node2 = q.submit([&](sycl::handler &cgh) {
      cgh.depends_on(Node1);
      // Internalization specified on each accessor.
      auto accTmp1 = bTmp1.get_access(
          cgh,
          sycl::property_list{sycl_ext::access_scope_work_item,
                              sycl_ext::fusion_internal_memory, sycl::no_init});
      auto accIn3 = bIn3.get_access(cgh);
      auto accTmp2 = bTmp2.get_access(cgh);
      cgh.parallel_for<class KernelOneOne>(
          Size, [=](sycl::id<1> i) { accTmp2[i] = accTmp1[i] * accIn3[i]; });
    });

    auto Node3 = q.submit([&](sycl::handler &cgh) {
      cgh.depends_on(Node1);
      // Internalization specified on each accessor.
      auto accTmp1 = bTmp1.get_access(
          cgh,
          sycl::property_list{sycl_ext::access_scope_work_item,
                              sycl_ext::fusion_internal_memory, sycl::no_init});
      auto accTmp3 = bTmp3.get_access(cgh);
      cgh.parallel_for<class KernelTwo>(
          Size, [=](sycl::id<1> i) { accTmp3[i] = accTmp1[i] * 5; });
    });

    auto Node4 = q.submit([&](sycl::handler &cgh) {
      cgh.depends_on({Node2, Node3});
      auto accTmp2 = bTmp2.get_access(cgh);
      auto accTmp3 = bTmp3.get_access(cgh);
      auto accOut = bOut.get_access(cgh);
      cgh.parallel_for<AddKernel>(Size, AddKernel{accTmp2, accTmp3, accOut});
    });

    graph.end_recording();

    // Trigger fusion during finalization.
    auto exec_graph =
        graph.finalize({sycl_ext::property::graph::require_fusion{}});

    q.ext_oneapi_graph(exec_graph);

    q.wait();
  }

  sycl::host_accessor HostAccOut(bOut);
  sycl::host_accessor HostAccTmp1(bTmp1);
  sycl::host_accessor HostAccTmp2(bTmp2);
  sycl::host_accessor HostAccTmp3(bTmp3);
  // Check the results
  for (size_t i = 0; i < Size; ++i) {
    assert(HostAccOut[i] == (ref[i]) && "Computation error");
    assert((HostAccTmp1[i] == zeros[i]) && "Not internalizing");
    assert((HostAccTmp2[i] == zeros[i]) && "Not internalizing");
    assert((HostAccTmp3[i] == zeros[i]) && "Not internalizing");
  }

  return 0;
}
