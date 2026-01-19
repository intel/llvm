// Tests using a raw_kernel_arg with 32-bit sized scalars.

#include "../graph_common.hpp"

auto constexpr CLSource = R"===(
__kernel void RawArgKernel(int scalar, __global int *out) {
  size_t id = get_global_id(0);
  out[id] = id + scalar;
}
)===";

int main() {
  queue Queue{};

  auto SourceKB =
      sycl::ext::oneapi::experimental::create_kernel_bundle_from_source(
          Queue.get_context(),
          sycl::ext::oneapi::experimental::source_language::opencl, CLSource);
  auto ExecKB = sycl::ext::oneapi::experimental::build(SourceKB);

  exp_ext::command_graph Graph{Queue};

  int32_t *Ptr = malloc_device<int32_t>(Size, Queue);
  Queue.memset(Ptr, 0, Size * sizeof(int32_t)).wait();

  int32_t Scalar = 42;
  exp_ext::raw_kernel_arg RawScalar(&Scalar, sizeof(int32_t));

  auto KernelNode = add_node(Graph, Queue, [&](handler &cgh) {
    cgh.set_arg(0, RawScalar);
    cgh.set_arg(1, Ptr);
    cgh.parallel_for(sycl::range<1>{Size},
                     ExecKB.ext_oneapi_get_kernel("RawArgKernel"));
  });

  auto ExecGraph = Graph.finalize();

  // Ptr should be filled with values based on Scalar
  Queue.ext_oneapi_graph(ExecGraph).wait();

  std::vector<int32_t> HostData(Size);
  Queue.copy(Ptr, HostData.data(), Size).wait();

  for (size_t i = 0; i < Size; i++) {
    assert(HostData[i] == (i + Scalar));
  }

  return 0;
}
