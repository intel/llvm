// REQUIRES: level_zero, level_zero_dev_kit
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %level_zero_options %s -o %t.ze.out
// RUN: env ONEAPI_DEVICE_SELECTOR="level_zero:*" %t.ze.out

#include <level_zero/ze_api.h>

#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>

constexpr auto BE = sycl::backend::ext_oneapi_level_zero;

int main() {
  sycl::device Dev{sycl::default_selector_v};

  sycl::queue Q{Dev};

  if (0) {
    Q.submit([](sycl::handler &CGH) { CGH.single_task<class T>([] {}); });
  }

  sycl::platform Plt = Dev.get_platform();
  auto NativePlt = sycl::get_native<BE>(Plt);

  sycl::platform NewPlt = sycl::make_platform<BE>(NativePlt);
  assert(NewPlt == Plt);

  return 0;
}
