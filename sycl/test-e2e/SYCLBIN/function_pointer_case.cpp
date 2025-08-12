
// REQUIRES: aspect-usm_device_allocations, aspect-usm_shared_allocations

// ptxas currently fails to compile images with unresolved symbols. Disable for
// other targets than SPIR-V until this has been resolved. (CMPLRLLVM-68810)
// Note: %{sycl_target_opts} should be added to the SYCLBIN compilation lines
// once fixed.
// REQUIRES: target-spir

// RUN: %clangxx --offload-new-driver -fsyclbin=input -fsycl-allow-device-image-dependencies -DSYCLBIN_INPUT %s -o %t.input.syclbin
// RUN: %clangxx --offload-new-driver -fsyclbin=object -fsycl-allow-device-image-dependencies -DSYCLBIN_OBJECT -Xclang -fsycl-allow-func-ptr %s -o %t.object.syclbin
// RUN: %{build} -o %t.out
//
// RUN: %{l0_leak_check} %{run} %t.out %t.input.syclbin %t.object.syclbin
//
// TODO: Add the following options to the object case once linking is supported
//       for AOT binaries:
//       -fgpu-rdc -fsycl-targets=... --offload-arch=...

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/syclbin_kernel_bundle.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

typedef void (*FuncPtrT)(size_t *);

struct ArgsT {
  size_t *Ptr;
  FuncPtrT FuncPtr;
};

#if defined(SYCLBIN_INPUT)

SYCL_EXTERNAL size_t GetID();

SYCL_EXTERNAL void Func(size_t *Ptr) {
  size_t GlobalID = GetID();
  Ptr[GlobalID] = GlobalID;
}

extern "C" SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (syclexp::single_task_kernel)) void GetFuncPtr(ArgsT *Args) {
  Args->FuncPtr = Func;
}

#elif defined(SYCLBIN_OBJECT)

SYCL_EXTERNAL size_t GetID() {
  return syclext::this_work_item::get_nd_item<1>().get_global_id();
}

extern "C" SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (syclexp::nd_range_kernel<1>)) void Kernel(ArgsT *Args) {
  (*Args->FuncPtr)(Args->Ptr);
}

#else

constexpr size_t N = 32;

int main(int argc, char *argv[]) {
  assert(argc == 3);

  sycl::queue Q;

  std::cout << "Load input SYCLBIN and compile it to object state."
            << std::endl;
  auto SYCLBINInput = syclexp::get_kernel_bundle<sycl::bundle_state::input>(
      Q.get_context(), std::string{argv[1]});
  auto SYCLBINInputObj = sycl::compile(SYCLBINInput);

  std::cout << "Load object SYCLBIN." << std::endl;
  auto SYCLBINObj = syclexp::get_kernel_bundle<sycl::bundle_state::object>(
      Q.get_context(), std::string{argv[2]});

  std::cout << "Link objects." << std::endl;
  auto KBExe = sycl::link({SYCLBINInputObj, SYCLBINObj});

  ArgsT *Args = sycl::malloc_shared<ArgsT>(N, Q);
  Args->Ptr = sycl::malloc_shared<size_t>(N, Q);

  // Prefetch the data pointer on the device. This is needed as the device
  // compiler might not be able to detect the use of the pointer, due to the
  // indirect call to the pointer-user function.
  // Though this is done prior to the GetFuncPtrKern launch, as it may avoid the
  // need for the copy-back of Args. The resulting event is only a dependency of
  // the launch of Kernel however.
  sycl::event ArgPtrPrefetchEvent = Q.prefetch(Args->Ptr, sizeof(ArgsT));

  // Get function pointer through kernel. This deviates from the original.
  sycl::kernel GetFuncPtrKern = KBExe.ext_oneapi_get_kernel("GetFuncPtr");
  std::cout << "Launching GetFuncPtr" << std::endl;
  Q.submit([&](sycl::handler &CGH) {
     CGH.set_args(Args);
     CGH.single_task(GetFuncPtrKern);
   }).wait();

  // Launch kernel.
  sycl::kernel Kern = KBExe.ext_oneapi_get_kernel("Kernel");
  std::cout << "Launching Kernel" << std::endl;
  Q.submit([&](sycl::handler &CGH) {
     CGH.depends_on(ArgPtrPrefetchEvent);
     CGH.set_args(Args);
     CGH.parallel_for(sycl::nd_range{{N}, {N}}, Kern);
   }).wait();

  int Failed = 0;
  for (size_t I = 0; I < N; ++I) {
    if (Args->Ptr[I] != I) {
      std::cout << Args->Ptr[I] << " != " << I << std::endl;
      ++Failed;
    }
  }

  if (!Failed)
    std::cout << "Results are a-okay!" << std::endl;

  sycl::free(Args->Ptr, Q);
  sycl::free(Args, Q);

  return Failed;
}

#endif
