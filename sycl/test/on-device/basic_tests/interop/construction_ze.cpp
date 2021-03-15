// REQUIRES: level_zero
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.ze.out
// RUN: %t.ze.out

#include <sycl/sycl.hpp>
#include <sycl/backend/level_zero.hpp>

constexpr auto BE = sycl::backend::level_zero;

int main() {
  sycl::device Dev{sycl::default_selector{}};
  sycl::context Ctx{Dev};

  sycl::platform Plt = Dev.get_platform();
  auto NativePlt = Plt.get_native<BE>();

  sycl::platform NewPlt = sycl::make_platform<BE>(NativePlt);
  assert(NewPlt == Plt);

  sycl::queue Q{Ctx, Dev};

  sycl::event Evt = Q.single_task<class Tst>([]{});
  auto NativeEvt = Evt.get_native<BE>();

  sycl::event NewEvt = sycl::make_event<BE>(NativeEvt, Ctx);
  assert(NewEvt == Evt);

  return 0;
}

