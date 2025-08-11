#include "common.hpp"

#include <sycl/usm.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

typedef void (*FuncPtrT)(size_t *);

struct ArgsT {
  size_t *Ptr;
  FuncPtrT *FuncPtr;
};

#ifdef __SYCL_DEVICE_ONLY__
SYCL_EXTERNAL size_t GetID();
#else
// Host-side code to avoid linker problems. Will never be called.
SYCL_EXTERNAL size_t GetID() { return 0; }
#endif

SYCL_EXTERNAL
void Func(size_t *Ptr) {
  size_t GlobalID = GetID();
  Ptr[GlobalID] = GlobalID;
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void GetFuncPtr(ArgsT Args) { *Args.FuncPtr = Func; }

constexpr size_t N = 32;

int main(int argc, char *argv[]) {
  assert(argc == 2);

  sycl::queue Q;

  int Failed = CommonLoadCheck(Q.get_context(), argv[1]);

#if defined(SYCLBIN_INPUT_STATE)
  auto SYCLBINInput = syclexp::get_kernel_bundle<sycl::bundle_state::input>(
      Q.get_context(), std::string{argv[1]});
  auto SYCLBINObj = sycl::compile(SYCLBINInput);
#elif defined(SYCLBIN_OBJECT_STATE)
  auto SYCLBINObj = syclexp::get_kernel_bundle<sycl::bundle_state::object>(
      Q.get_context(), std::string{argv[1]});
#else // defined(SYCLBIN_EXECUTABLE_STATE)
#error "Test does not work with executable state."
#endif

  auto KBObj =
      syclexp::get_kernel_bundle<GetFuncPtr, sycl::bundle_state::object>(
          Q.get_context());
  auto KBExe = sycl::link({KBObj, SYCLBINObj});

  ArgsT Args{};
  Args.FuncPtr = sycl::malloc_shared<FuncPtrT>(N, Q);
  Args.Ptr = sycl::malloc_shared<size_t>(N, Q);

  sycl::kernel GetFuncPtrKern = KBExe.ext_oneapi_get_kernel<GetFuncPtr>();
  Q.submit([&](sycl::handler &CGH) {
     CGH.set_args(Args);
     CGH.single_task(GetFuncPtrKern);
   }).wait();

  sycl::kernel Kern = KBExe.ext_oneapi_get_kernel("Kernel");
  Q.submit([&](sycl::handler &CGH) {
     CGH.set_args(Args);
     CGH.parallel_for(sycl::nd_range{{N}, {N}}, Kern);
   }).wait();

  for (size_t I = 0; I < N; ++I) {
    if (Args.Ptr[I] != I) {
      std::cout << Args.Ptr[I] << " != " << I << std::endl;
      ++Failed;
    }
  }

  sycl::free(Args.FuncPtr, Q);
  sycl::free(Args.Ptr, Q);

  return Failed;
}
