// REQUIRES: opencl, opencl_icd
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %opencl_lib %s -o %t.ocl.out
// RUN: env SYCL_DEVICE_FILTER="opencl" %t.ocl.out

#include <CL/cl.h>
#include <CL/sycl/backend/opencl.hpp>

#include <sycl/sycl.hpp>

constexpr auto BE = sycl::backend::opencl;

int main() {
  sycl::device Dev{sycl::default_selector{}};
  auto NativeDev = Dev.get_native<BE>();

  sycl::device NewDev = sycl::make_device<BE>(NativeDev);
  assert(NewDev.get_info<sycl::info::device::name>() ==
         Dev.get_info<sycl::info::device::name>());

  sycl::platform Plt = Dev.get_platform();
  auto NativePlt = Plt.get_native<BE>();

  sycl::platform NewPlt = sycl::make_platform<BE>(NativePlt);
  assert(NewPlt == Plt);

  sycl::context Ctx{Dev};
  auto NativeCtx = Ctx.get_native<BE>();

  sycl::context NewCtx = sycl::make_context<BE>(NativeCtx);
  assert(NewCtx.get_native<BE>() == NativeCtx);

  sycl::queue Q{Ctx, Dev};
  auto NativeQ = Q.get_native<BE>();

  sycl::queue NewQ = sycl::make_queue<BE>(NativeQ, Ctx);
  assert(NativeQ == NewQ.get_native<BE>());

  sycl::event Evt = Q.single_task<class Tst>([]{});
  auto NativeEvt = Evt.get_native<BE>();

  sycl::event NewEvt = sycl::make_event<BE>(NativeEvt, Ctx);
  assert(NativeEvt == NewEvt.get_native<BE>());

  cl_mem NativeBuf =
      clCreateBuffer(NativeCtx, CL_MEM_READ_WRITE, 128, nullptr, nullptr);
  auto NewBuf = sycl::make_buffer<BE, char>(NativeBuf, Ctx);
  assert(NewBuf.get_range()[0] == 128);

  constexpr const char *ProgSrc = "kernel _() {}";
  cl_error Err;

  cl_program OclProg =
      clCreateProgramWithSource(NativeCtx, 1, &ProgSrc, nullptr, &Err);
  if (Err != CL_SUCCESS)
    return -1;

  auto KB = sycl::make_kernel_bundle<BE, bundle_state::input>(OclProg, Ctx);
  assert(KB.has_kernel("_"));

  Err = clBuildProgram(OclProg, 1, &NativeDev, "", nullptr, nullptr);
  assert(Err == CL_SUCCESS);

  cl_kernel NativeKer = clCreateKernel(OclProg, "_", &err);
  assert(Err == CL_SUCCESS);

  auto Kernel = sycl::make_kernel<BE>(NativeKer, Ctx);
  assert(Kernel.get_info<sycl::info::kernel::name>() == "_");

  return 0;
}
