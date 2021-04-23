// REQUIRES: level_zero, level_zero_dev_kit
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %level_zero_options %s -o %t.ze.out
// RUN: env SYCL_DEVICE_FILTER="level_zero" %t.ze.out

#include <level_zero/ze_api.h>

#include <CL/sycl/backend/level_zero.hpp>
#include <sycl/sycl.hpp>

#include <CL/sycl/INTEL/online_compiler.hpp>

constexpr auto BE = sycl::backend::level_zero;

int main() {
  sycl::device Dev{sycl::default_selector{}};

  sycl::context Ctx{Dev};
  sycl::queue Q{Ctx, Dev};

  if (0) {
    Q.submit([](sycl::handler &CGH) { CGH.single_task<class T>([] {}); });
  }

  sycl::platform Plt = Dev.get_platform();
  auto NativePlt = Plt.get_native<BE>();

  sycl::platform NewPlt = sycl::make_platform<BE>(NativePlt);
  assert(NewPlt == Plt);

  {
    sycl::INTEL::online_compiler<sycl::INTEL::source_language::opencl_c>
        Compiler;

    constexpr const char *Source = "kernel void _() {}";

    std::vector<unsigned char> IL = Compiler.compile(Source);

    sycl::level_zero::module_desc_t ModuleDesc{
        nullptr, sycl::level_zero::module_desc_t::state::il, std::move(IL)};

    sycl::kernel_bundle KBInput =
        sycl::make_kernel_bundle<BE, sycl::bundle_state::input>(ModuleDesc,
                                                                Ctx);
    sycl::kernel_bundle KBObject =
        sycl::make_kernel_bundle<BE, sycl::bundle_state::object>(ModuleDesc,
                                                                 Ctx);
    sycl::kernel_bundle KBExe =
        sycl::make_kernel_bundle<BE, sycl::bundle_state::executable>(ModuleDesc,
                                                                     Ctx);
  }

  return 0;
}
