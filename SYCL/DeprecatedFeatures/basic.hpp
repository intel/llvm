#include <sycl/sycl.hpp>

int main(int argc, char **argv) {
  // Test program and kernel APIs when building a kernel.
  {
    sycl::queue q;
    int data = 0;
    {
      sycl::buffer<int, 1> buf(&data, sycl::range<1>(1));
      sycl::program prg1(q.get_context());
      sycl::program prg2(q.get_context());
      sycl::program prg3(q.get_context());
      sycl::program prg4(q.get_context());
      sycl::program prg5(q.get_context());

      prg1.build_with_kernel_type<class BuiltKernel>(); // 1 cache item
      prg2.build_with_kernel_type<class BuiltKernel>(
          "-cl-fast-relaxed-math"); // +1 cache item due to build options
      prg3.build_with_kernel_type<class BuiltKernel>();    // program binary is
                                                           // equal to prg1
      prg4.build_with_kernel_type<class CompiledKernel>(); // program binary is
                                                           // equal to prg1
      sycl::kernel krn = prg2.get_kernel<class BuiltKernel>();

      q.submit([&](sycl::handler &cgh) {
        auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
        cgh.single_task<class BuiltKernel>(krn, [=]() { acc[0] = acc[0] + 1; });
      });
    }
    assert(data == 1);
  }

  // Test program and kernel APIs when compiling / linking a kernel.
  {
    sycl::queue q;
    int data = 0;
    {
      sycl::buffer<int, 1> buf(&data, sycl::range<1>(1));
      sycl::program prg6(q.get_context());
      sycl::program prg7(q.get_context());
      sycl::program prg8(q.get_context());
      prg6.compile_with_kernel_type<class CompiledKernel>();
      prg6.link(); // The binary is not cached for separate compile/link
      prg7.build_with_kernel_type<class CompiledKernel>(
          "-cl-fast-relaxed-math"); // program binary is equal to prg2
      prg8.build_with_kernel_type<class CompiledKernel>(
          "-g"); // +1 cache item due to build options

      sycl::kernel krn = prg6.get_kernel<class CompiledKernel>();

      q.submit([&](sycl::handler &cgh) {
        auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
        cgh.single_task<class CompiledKernel>(krn,
                                              [=]() { acc[0] = acc[0] + 1; });
      });
    }
    assert(data == 1);
  }
  return 0;
}
