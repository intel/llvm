// RUN: %{build} -o %t.out %threads_lib %opencl_lib
// RUN: %{run} %t.out
// RUN: %if preview-breaking-changes-supported %{ %{build} -fpreview-breaking-changes -o %t2.out %threads_lib %opencl_lib %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t2.out %}
// REQUIRES: opencl, opencl_icd

#include <sycl/backend.hpp>
#include <sycl/detail/cl.h>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/interop_handle.hpp>

namespace exp_ext = sycl::ext::oneapi::experimental;
using namespace sycl;

int main() {
  queue Queue;

  const size_t Size = 128;
  buffer<int, 1> BufX{Size};
  BufX.set_write_back(false);
  buffer<int, 1> BufY{Size};
  BufY.set_write_back(false);

  {
    exp_ext::command_graph Graph{
        Queue, {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

    Graph.add([&](handler &CGH) {
      auto AccX = BufX.get_access(CGH);
      auto AccY = BufY.get_access(CGH);
      CGH.single_task([=]() {
        for (size_t i = 0; i < Size; i++) {
          AccX[i] = i;
          AccY[i] = 0;
        }
      });
    });

    auto Platform =
        get_native<backend::opencl>(Queue.get_context().get_platform());
    clCommandCopyBufferKHR_fn clCommandCopyBufferKHR =
        reinterpret_cast<clCommandCopyBufferKHR_fn>(
            clGetExtensionFunctionAddressForPlatform(Platform,
                                                     "clCommandCopyBufferKHR"));
    assert(clCommandCopyBufferKHR != nullptr);

    Graph.add([&](handler &CGH) {
      auto AccX = BufX.get_access(CGH);
      auto AccY = BufY.get_access(CGH);

      CGH.ext_codeplay_enqueue_native_command([=](interop_handle IH) {
        if (!IH.ext_codeplay_has_graph()) {
          assert(false && "Native Handle should have a graph");
        }
        cl_command_buffer_khr NativeGraph =
            IH.ext_codeplay_get_native_graph<backend::opencl>();
        auto SrcBuffer = IH.get_native_mem<backend::opencl>(AccX);
        auto DstBuffer = IH.get_native_mem<backend::opencl>(AccY);

        auto Res = clCommandCopyBufferKHR(
            NativeGraph, nullptr, nullptr, SrcBuffer[0], DstBuffer[0], 0, 0,
            Size * sizeof(int), 0, nullptr, nullptr, nullptr);
        assert(Res == CL_SUCCESS);
      });
    });

    Graph.add([&](handler &CGH) {
      auto AccY = BufY.get_access(CGH);
      CGH.parallel_for(range<1>{Size}, [=](id<1> it) { AccY[it] *= 2; });
    });

    auto ExecGraph = Graph.finalize();
    Queue.ext_oneapi_graph(ExecGraph).wait();
  }

  auto HostAcc = BufY.get_host_access();
  for (size_t i = 0; i < Size; i++) {
    const int Ref = i * 2;
    assert(Ref == HostAcc[i]);
  }

  return 0;
}
