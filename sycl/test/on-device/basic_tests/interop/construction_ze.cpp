// REQUIRES: level_zero
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.ze.out
// RUN: %t.ze.out

#include <level_zero/ze_api.h>

#include <CL/sycl/backend/level_zero.hpp>
#include <sycl/sycl.hpp>

constexpr auto BE = sycl::backend::level_zero;

int main() {
  sycl::device Dev{sycl::default_selector{}};

  sycl::platform Plt = Dev.get_platform();
  auto NativePlt = Plt.get_native<BE>();

  sycl::platform NewPlt = sycl::make_platform<BE>(NativePlt);
  assert(NewPlt == Plt);

  // TODO uncomment once events are supported in L0 backend interop.
  /*
    sycl::context Ctx{Dev};
    sycl::queue Q{Ctx, Dev};

    sycl::event Evt = Q.single_task<class Tst>([]{});
    auto NativeEvt = Evt.get_native<BE>();

    sycl::event NewEvt = sycl::make_event<BE>(NativeEvt, Ctx);
    assert(NativeEvt == NewEvt.get_native<BE>());
  */

  return 0;
}
