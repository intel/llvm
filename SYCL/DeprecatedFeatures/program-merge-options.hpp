#define SYCL2020_DISABLE_DEPRECATION_WARNINGS

#include <CL/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

class KernelName;
void submitKernel() {
  cl::sycl::queue q;
  q.submit([&](cl::sycl::handler &cgh) {
    cgh.single_task<KernelName>([]() SYCL_ESIMD_KERNEL {});
  });
}

int main() {
  const std::string CompileOpts{"-DCOMPILE_OPTS"};
  const std::string LinkOpts{"-cl-fast-relaxed-math"};
  const std::string BuildOpts{"-DBUILD_OPTS"};

  try {
    cl::sycl::context Ctx;
    cl::sycl::program PrgA{Ctx};

    PrgA.build_with_kernel_type<KernelName>(BuildOpts);

    cl::sycl::program PrgB{Ctx};
    PrgB.compile_with_kernel_type<KernelName>(CompileOpts);

    PrgB.link(LinkOpts);
  } catch (...) {
    // Ignore all exceptions
  }
}
