// REQUIRES: opencl, opencl_icd
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %opencl_lib %s -o %t.ocl.out
// RUN: env SYCL_DEVICE_FILTER="opencl" %t.ocl.out

#include <CL/cl.h>
#include <sycl/backend/opencl.hpp>

#include <sycl/sycl.hpp>

constexpr auto BE = sycl::backend::opencl;

int main() {
  sycl::device Dev{sycl::default_selector_v};
  auto NativeDev = sycl::get_native<BE>(Dev);

  sycl::device NewDev = sycl::make_device<BE>(NativeDev);
  assert(NewDev.get_info<sycl::info::device::name>() ==
         Dev.get_info<sycl::info::device::name>());

  sycl::platform Plt = Dev.get_platform();
  auto NativePlt = sycl::get_native<BE>(Plt);

  sycl::platform NewPlt = sycl::make_platform<BE>(NativePlt);
  assert(NewPlt == Plt);

  sycl::context Ctx{Dev};
  auto NativeCtx = sycl::get_native<BE>(Ctx);

  sycl::context NewCtx = sycl::make_context<BE>(NativeCtx);
  assert(sycl::get_native<BE>(NewCtx) == NativeCtx);

  sycl::queue Q{Ctx, Dev};
  auto NativeQ = sycl::get_native<BE>(Q);

  sycl::queue NewQ = sycl::make_queue<BE>(NativeQ, Ctx);
  assert(NativeQ == sycl::get_native<BE>(NewQ));

  sycl::event Evt = Q.single_task<class Tst>([] {});
  auto NativeEvt = sycl::get_native<BE>(Evt);

  sycl::event NewEvt = sycl::make_event<BE>(NativeEvt, Ctx);
  assert(NativeEvt == sycl::get_native<BE>(NewEvt));

  cl_mem NativeBuf =
      clCreateBuffer(NativeCtx, CL_MEM_READ_WRITE, 128, nullptr, nullptr);
  auto NewBuf = sycl::make_buffer<BE, char>(NativeBuf, Ctx);
  assert(NewBuf.get_range()[0] == 128);

  const char *ProgSrc = "kernel void _() {}";
  cl_int Err;

  // Program in state NONE
  {
    cl_program OclProg =
        clCreateProgramWithSource(NativeCtx, 1, &ProgSrc, nullptr, &Err);
    assert(Err == CL_SUCCESS && "Program creation failed");

    auto KB =
        sycl::make_kernel_bundle<BE, sycl::bundle_state::input>(OclProg, Ctx);
    auto KernelIDs = KB.get_kernel_ids();
    assert(KernelIDs.empty());

    cl_program OclProg2 =
        clCreateProgramWithSource(NativeCtx, 1, &ProgSrc, nullptr, &Err);
    assert(Err == CL_SUCCESS && "Program creation failed");

    auto KB2 =
        sycl::make_kernel_bundle<BE, sycl::bundle_state::object>(OclProg2, Ctx);
    auto KernelIDs2 = KB2.get_kernel_ids();
    assert(KernelIDs2.empty());

    cl_program OclProg3 =
        clCreateProgramWithSource(NativeCtx, 1, &ProgSrc, nullptr, &Err);
    assert(Err == CL_SUCCESS && "Program creation failed");

    auto KB3 = sycl::make_kernel_bundle<BE, sycl::bundle_state::executable>(
        OclProg3, Ctx);
    auto KernelIDs3 = KB3.get_kernel_ids();
    assert(KernelIDs3.empty());
  }

  // Compiled program
  {
    cl_program OclProg =
        clCreateProgramWithSource(NativeCtx, 1, &ProgSrc, nullptr, &Err);
    assert(Err == CL_SUCCESS && "Program creation failed");

    Err = clCompileProgram(OclProg, 1, &NativeDev, "", 0, nullptr, nullptr,
                           nullptr, nullptr);
    assert(Err == CL_SUCCESS && "Program compile failed");

    auto KB =
        sycl::make_kernel_bundle<BE, sycl::bundle_state::object>(OclProg, Ctx);
    auto KernelIDs = KB.get_kernel_ids();
    assert(KernelIDs.empty());

    bool StateMismatch = false;
    try {
      auto KB2 =
          sycl::make_kernel_bundle<BE, sycl::bundle_state::input>(OclProg, Ctx);
    } catch (sycl::runtime_error Ex) {
      StateMismatch = true;
    }
    assert(StateMismatch);

    cl_program OclProg3 =
        clCreateProgramWithSource(NativeCtx, 1, &ProgSrc, nullptr, &Err);
    assert(Err == CL_SUCCESS && "Program creation failed");

    Err = clCompileProgram(OclProg3, 1, &NativeDev, "", 0, nullptr, nullptr,
                           nullptr, nullptr);
    assert(Err == CL_SUCCESS && "Program compile failed");

    auto KB3 = sycl::make_kernel_bundle<BE, sycl::bundle_state::executable>(
        OclProg3, Ctx);
    auto KernelIDs3 = KB3.get_kernel_ids();
    assert(KernelIDs3.empty());
  }

  // Linked program
  {
    cl_program OclProg =
        clCreateProgramWithSource(NativeCtx, 1, &ProgSrc, nullptr, &Err);
    assert(Err == CL_SUCCESS && "Program creation failed");

    Err = clBuildProgram(OclProg, 1, &NativeDev, "", nullptr, nullptr);
    assert(Err == CL_SUCCESS && "Program build failed");

    auto KB = sycl::make_kernel_bundle<BE, sycl::bundle_state::executable>(
        OclProg, Ctx);
    auto KernelIDs = KB.get_kernel_ids();
    assert(KernelIDs.empty());

    cl_kernel NativeKer = clCreateKernel(OclProg, "_", &Err);
    assert(Err == CL_SUCCESS && "Kernel creation failed");

    auto Kernel = sycl::make_kernel<BE>(NativeKer, Ctx);
    assert(Kernel.get_info<sycl::info::kernel::num_args>() == 0);

    bool StateMismatch = false;
    try {
      auto KB2 =
          sycl::make_kernel_bundle<BE, sycl::bundle_state::input>(OclProg, Ctx);
    } catch (sycl::runtime_error Ex) {
      StateMismatch = true;
    }
    assert(StateMismatch);

    StateMismatch = false;
    try {
      auto KB3 = sycl::make_kernel_bundle<BE, sycl::bundle_state::object>(
          OclProg, Ctx);
    } catch (sycl::runtime_error Ex) {
      StateMismatch = true;
    }
    assert(StateMismatch);
  }

  return 0;
}
