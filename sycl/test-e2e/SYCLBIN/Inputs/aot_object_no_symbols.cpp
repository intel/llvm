// SYCLBIN producer for a fully self-contained object-state artifact: the
// kernel neither imports nor exports any SYCL_EXTERNAL symbol, so
// sycl-post-link emits no "SYCL/imported symbols" and no
// "SYCL/exported symbols" property set for it.
//
// When compiled AOT, ProgramManager::getBinImageState therefore classifies
// the resulting native device code image as bundle_state::executable, even
// though the SYCLBIN's global metadata records object state. This is the
// producer side of the "a .o with no undefined symbols is still a .o" case;
// the consumer is ../aot_object_no_symbols.cpp.

#include <sycl/sycl.hpp>

// No SYCL_EXTERNAL on the kernel: it is defined in this translation unit and
// only ever reached through the SYCLBIN, so nothing needs to be exported.
// extern "C" avoids name mangling so the kernel can be looked up by name.
extern "C" {
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (sycl::ext::oneapi::experimental::single_task_kernel))
void TestKernelNoSyms(int *Ptr, int Size) {
  for (int I = 0; I < Size; ++I)
    Ptr[I] = I;
}
}
