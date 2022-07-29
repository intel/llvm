#define SYCL2020_DISABLE_DEPRECATION_WARNINGS

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

class KernelName;
void submitKernel() {
  sycl::queue q;
  q.submit([&](sycl::handler &cgh) {
    cgh.single_task<KernelName>([]() SYCL_ESIMD_KERNEL {});
  });
}

int main() {
  const std::string CompileOpts{"-DCOMPILE_OPTS"};
  const std::string LinkOpts{"-cl-fast-relaxed-math"};
  const std::string BuildOpts{"-DBUILD_OPTS"};

  try {
    sycl::context Ctx;
    sycl::program PrgA{Ctx};

    PrgA.build_with_kernel_type<KernelName>(BuildOpts);

    sycl::program PrgB{Ctx};
    PrgB.compile_with_kernel_type<KernelName>(CompileOpts);

    PrgB.link(LinkOpts);
  } catch (...) {
    // Ignore all exceptions
  }
}
