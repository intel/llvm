#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/experimental/online_compiler.hpp>

#include <iostream>
#include <vector>

auto constexpr CLSource = R"===(
__kernel void my_kernel(__global int *in, __global int *out) {
  size_t i = get_global_id(0);
  out[i] = in[i]*2 + 100;
}
)===";

auto constexpr CLSourceSyntaxError = R"===(
__kernel void my_kernel(__global int *in, __global int *out) {
  syntax error here
  size_t i = get_global_id(0);
  out[i] = in[i]*2 + 100;
}
)===";

auto constexpr CMSource = R"===(
extern "C"
void cm_kernel() {
}
)===";

using namespace sycl::ext::intel;

#ifdef RUN_KERNELS
void testSyclKernel(sycl::queue &Q, sycl::kernel Kernel) {
  std::cout << "Run the kernel now:\n";
  constexpr int N = 4;
  int InputArray[N] = {0, 1, 2, 3};
  int OutputArray[N] = {};

  sycl::buffer<int, 1> InputBuf(InputArray, sycl::range<1>(N));
  sycl::buffer<int, 1> OutputBuf(OutputArray, sycl::range<1>(N));

  Q.submit([&](sycl::handler &CGH) {
    CGH.set_arg(0, InputBuf.get_access<sycl::access::mode::read>(CGH));
    CGH.set_arg(1, OutputBuf.get_access<sycl::access::mode::write>(CGH));
    CGH.parallel_for(sycl::range<1>{N}, Kernel);
  });

  auto Out = OutputBuf.get_host_access();
  for (int I = 0; I < N; I++)
    std::cout << I << "*2 + 100 = " << Out[I] << "\n";
}
#endif // RUN_KERNELS

int main(int argc, char **argv) {
  sycl::queue Q;
  sycl::device Device = Q.get_device();

  { // Compile and run a trivial OpenCL kernel.
    std::cout << "Test case1\n";
    sycl::ext::intel::experimental::online_compiler<
        sycl::ext::intel::experimental::source_language::opencl_c>
        Compiler;
    std::vector<byte> IL;
    try {
      IL = Compiler.compile(
          CLSource,
          // Pass two options to check that more than one is accepted.
          std::vector<std::string>{"-cl-fast-relaxed-math",
                                   "-cl-finite-math-only"});
      std::cout << "IL size = " << IL.size() << "\n";
      assert(IL.size() > 0 && "Unexpected IL size");
    } catch (sycl::exception &e) {
      std::cout << "Compilation to IL failed: " << e.what() << "\n";
      return 1;
    }
#ifdef RUN_KERNELS
    testSyclKernel(Q, getSYCLKernelWithIL(Q, IL));
#endif // RUN_KERNELS
  }

  { // Compile and run a trivial OpenCL kernel using online_compiler()
    // constructor accepting SYCL device.
    std::cout << "Test case2\n";
    sycl::ext::intel::experimental::online_compiler<
        sycl::ext::intel::experimental::source_language::opencl_c>
        Compiler(Device);
    std::vector<byte> IL;
    try {
      IL = Compiler.compile(CLSource);
      std::cout << "IL size = " << IL.size() << "\n";
      assert(IL.size() > 0 && "Unexpected IL size");
    } catch (sycl::exception &e) {
      std::cout << "Compilation to IL failed: " << e.what() << "\n";
      return 1;
    }
#ifdef RUN_KERNELS
    testSyclKernel(Q, getSYCLKernelWithIL(Q, IL));
#endif // RUN_KERNELS
  }

  // TODO: this test is temporarily turned off because CI buildbots do not set
  // PATHs to clangFEWrapper library properly.
  { // Compile a trivial CM kernel.
    std::cout << "Test case3\n";
    sycl::ext::intel::experimental::online_compiler<
        sycl::ext::intel::experimental::source_language::cm>
        Compiler;
    try {
      std::vector<byte> IL = Compiler.compile(CMSource);

      std::cout << "IL size = " << IL.size() << "\n";
      assert(IL.size() > 0 && "Unexpected IL size");
    } catch (sycl::exception &e) {
      std::cout << "Compilation to IL failed: " << e.what() << "\n";
      return 1;
    }
  }

  { // Compile a source with syntax errors.
    std::cout << "Test case4\n";
    sycl::ext::intel::experimental::online_compiler<
        sycl::ext::intel::experimental::source_language::opencl_c>
        Compiler;
    std::vector<byte> IL;
    bool TestPassed = false;
    try {
      IL = Compiler.compile(CLSourceSyntaxError);
    } catch (sycl::exception &e) {
      std::string Msg = e.what();
      if (Msg.find("syntax error here") != std::string::npos)
        TestPassed = true;
      else
        std::cerr << "Unexpected exception: " << Msg << "\n";
    }
    assert(TestPassed && "Failed to throw an exception for syntax error");
    if (!TestPassed)
      return 1;
  }

  { // Compile a good CL source using unrecognized compilation options.
    std::cout << "Test case5\n";
    sycl::ext::intel::experimental::online_compiler<
        sycl::ext::intel::experimental::source_language::opencl_c>
        Compiler;
    std::vector<byte> IL;
    bool TestPassed = false;
    try {
      IL = Compiler.compile(CLSource,
                            // Intentionally use incorrect option.
                            std::vector<std::string>{"WRONG_OPTION"});
    } catch (sycl::exception &e) {
      std::string Msg = e.what();
      if (Msg.find("WRONG_OPTION") != std::string::npos)
        TestPassed = true;
      else
        std::cerr << "Unexpected exception: " << Msg << "\n";
    }
    assert(TestPassed &&
           "Failed to throw an exception for unrecognized option");
    if (!TestPassed)
      return 1;
  }

  { // Try compiling CM source with OpenCL compiler.
    std::cout << "Test case6\n";
    sycl::ext::intel::experimental::online_compiler<
        sycl::ext::intel::experimental::source_language::opencl_c>
        Compiler;
    std::vector<byte> IL;
    bool TestPassed = false;
    try {
      // Intentionally pass CMSource instead of CLSource.
      IL = Compiler.compile(CMSource);
    } catch (sycl::exception &e) {
      std::string Msg = e.what();
      if (Msg.find("error: expected identifier or '('") != std::string::npos)
        TestPassed = true;
      else
        std::cerr << "Unexpected exception: " << Msg << "\n";
    }
    assert(TestPassed && "Failed to throw an exception for wrong program");
    if (!TestPassed)
      return 1;
  }

  std::cout << "\nAll test cases passed.\n";
  return 0;
}
