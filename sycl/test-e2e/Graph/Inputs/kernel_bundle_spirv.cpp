// Tests creating a node using a SPIR-V kernel imported with
// sycl_ext_oneapi_kernel_compiler_spirv. The SPIR-V kernels used in this test
// are identical to the ones used in KernelCompiler/Kernels/kernels.spv

#include "../graph_common.hpp"

int main(int, char **argv) {
  const sycl::device Dev{sycl::default_selector_v};
  const sycl::context Ctx{Dev};

  queue Queue{Ctx, Dev};

  sycl::kernel_bundle KernelBundle = loadKernelsFromFile(Queue, argv[1]);
  const auto getKernel =
      [](sycl::kernel_bundle<sycl::bundle_state::executable> &bundle,
         const std::string &name) {
        return bundle.ext_oneapi_get_kernel(name);
      };

  sycl::kernel kernel = getKernel(KernelBundle, "my_kernel");
  assert(kernel.get_backend() == backend::ext_oneapi_level_zero);

  constexpr int N = 4;
  std::array<int, N> input_array{0, 1, 2, 3};
  std::array<int, N> output_array{};
  std::array<int, N> output_array2{};

  sycl::buffer input_buffer(input_array.data(), sycl::range<1>(N));
  sycl::buffer output_buffer(output_array.data(), sycl::range<1>(N));
  sycl::buffer output_buffer2(output_array2.data(), sycl::range<1>(N));

  input_buffer.set_write_back(false);
  output_buffer.set_write_back(false);
  output_buffer2.set_write_back(false);

  {
    exp_ext::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

    add_node(Graph, Queue, ([&](sycl::handler &CGH) {
               CGH.set_arg(
                   0, input_buffer.get_access<sycl::access::mode::read>(CGH));
               CGH.set_arg(
                   1, output_buffer.get_access<sycl::access::mode::write>(CGH));
               CGH.parallel_for(sycl::range<1>{N}, kernel);
             }));

    add_node(Graph, Queue, ([&](sycl::handler &CGH) {
               CGH.set_arg(
                   0, input_buffer.get_access<sycl::access::mode::read>(CGH));
               CGH.set_arg(
                   1,
                   output_buffer2.get_access<sycl::access::mode::write>(CGH));
               CGH.parallel_for(sycl::range<1>{N}, kernel);
             }));

    auto GraphExec = Graph.finalize();

    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
    Queue.wait_and_throw();
  }

  host_accessor HostAccOutput(output_buffer);
  host_accessor HostAccOutput2(output_buffer2);

  for (int i = 0; i < N; i++) {
    assert(HostAccOutput[i] == ((i * 2) + 100));
    assert(HostAccOutput2[i] == ((i * 2) + 100));
  }

  return 0;
}
