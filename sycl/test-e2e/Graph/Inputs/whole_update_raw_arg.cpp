// Tests whole graph update with raw argument extensions

#include "../graph_common.hpp"

void SubmitKernelNode(
    exp_ext::command_graph<exp_ext::graph_state::modifiable> Graph, queue Queue,
    int32_t *Ptr, exp_ext::raw_kernel_arg &RawArg, kernel Kernel) {

  add_node(Graph, Queue, [&](handler &cgh) {
    cgh.set_arg(0, RawArg);
    cgh.set_arg(1, Ptr);
    cgh.parallel_for(sycl::range<1>{Size}, Kernel);
  });
}

int main() {
  queue Queue{};

  auto constexpr CLSource = R"===(
__kernel void RawArgKernel(int scalar, __global int *out) {
  size_t id = get_global_id(0);
  out[id] = id + scalar;
}
)===";

  auto SourceKB =
      sycl::ext::oneapi::experimental::create_kernel_bundle_from_source(
          Queue.get_context(),
          sycl::ext::oneapi::experimental::source_language::opencl, CLSource);
  auto Kernel =
      sycl::ext::oneapi::experimental::build(SourceKB).ext_oneapi_get_kernel(
          "RawArgKernel");

  exp_ext::command_graph GraphA{Queue};

  const size_t N = 1024;
  int32_t *PtrA = malloc_device<int32_t>(N, Queue);
  Queue.memset(PtrA, 0, N * sizeof(int32_t)).wait();

  int32_t ScalarA = 42;
  sycl::ext::oneapi::experimental::raw_kernel_arg RawScalarA(&ScalarA,
                                                             sizeof(int32_t));

  SubmitKernelNode(GraphA, Queue, PtrA, RawScalarA, Kernel);
  auto ExecGraphA = GraphA.finalize(exp_ext::property::graph::updatable{});

  // PtrA should be filled with values based on ScalarA
  Queue.ext_oneapi_graph(ExecGraphA).wait();

  std::vector<int32_t> HostDataA(N);
  Queue.copy(PtrA, HostDataA.data(), N).wait();
  for (size_t i = 0; i < N; i++) {
    assert(HostDataA[i] == (i + ScalarA));
  }

  exp_ext::command_graph GraphB{Queue.get_context(), Queue.get_device()};

  int32_t *PtrB = malloc_device<int32_t>(N, Queue);
  Queue.memset(PtrB, 0, N * sizeof(int32_t)).wait();

  int32_t ScalarB = 0xA;
  sycl::ext::oneapi::experimental::raw_kernel_arg RawScalarB(&ScalarB,
                                                             sizeof(int32_t));

  // Swap ScalarB and PtrB to be the new inputs/outputs
  SubmitKernelNode(GraphB, Queue, PtrB, RawScalarB, Kernel);
  ExecGraphA.update(GraphB);
  Queue.ext_oneapi_graph(ExecGraphA).wait();

  std::vector<int32_t> HostDataB(N);
  Queue.copy(PtrA, HostDataA.data(), N);
  Queue.copy(PtrB, HostDataB.data(), N);
  Queue.wait();
  for (size_t i = 0; i < N; i++) {
    assert(HostDataA[i] == (i + ScalarA));
    assert(HostDataB[i] == (i + ScalarB));
  }
  return 0;
}
