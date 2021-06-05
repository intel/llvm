// REQUIRES: level_zero, level_zero_dev_kit
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %level_zero_options %s -o %t.ze.out
// RUN: env SYCL_DEVICE_FILTER="level_zero" %t.ze.out

#include <level_zero/ze_api.h>

#include <CL/sycl/backend/level_zero.hpp>
#include <sycl/sycl.hpp>

constexpr auto BE = sycl::backend::level_zero;

int main() {
  sycl::device Dev{sycl::default_selector{}};

  sycl::queue Q{Dev};

  if (0) {
    Q.submit([](sycl::handler &CGH) { CGH.single_task<class T>([] {}); });
  }

  sycl::platform Plt = Dev.get_platform();
  auto NativePlt = Plt.get_native<BE>();

  sycl::platform NewPlt = sycl::make_platform<BE>(NativePlt);
  assert(NewPlt == Plt);

  return 0;
}
