// REQUIRES: opencl
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -lOpenCL %s -o %t.ocl.out
// RUN: %t.ocl.out

#include <sycl/sycl.hpp>
#include <sycl/backend/opencl.hpp>

constexpr auto BE = sycl::backend::opencl;

int main() {
  sycl::device Dev{sycl::default_selector{}};
  auto NativeDev = Dev.get_native<BE>();

  sycl::device NewDev = sycl::make_device<BE>(NativeDev);
  assert(NewDev == Dev);

  sycl::platform Plt = Dev.get_platform();
  auto NativePlt = Plt.get_native<BE>();

  sycl::platform NewPlt = sycl::make_platform<BE>(NativePlt);
  assert(NewPlt == Plt);

  sycl::context Ctx{Dev};
  auto NativeCtx = Ctx.get_native<BE>();

  sycl::context NewCtx = sycl::make_context<BE>(NativeCtx);
  assert(NewCtx == NativeCtx);

  sycl::queue Q{Ctx, Dev};
  auto NativeQ = Q.get_native<BE>();

  sycl::queue NewQ = sycl::make_queue<BE>(NativeQ, Ctx);
  assert(Q == NewQ);

  sycl::event Evt = Q.single_task<class Tst>([]{});
  auto NativeEvt = Evt.get_native<BE>();

  sycl::event NewEvt = sycl::make_event<BE>(NativeEvt, Ctx);
  assert(NewEvt == Evt);

  cl_mem NativeBuf =
      clCreateBuffer(NativeCtx, CL_MEM_READ_WRITE, 128, nullptr, nullptr);
  auto NewBuf = sycl::make_buffer<BE, char>(NativeBuf, Ctx);
  assert(NewBuf.get_range()[0] == 128);

  return 0;
}
