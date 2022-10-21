#pragma once

#include <sycl/detail/defines_elementary.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::intel::experimental {

enum class architecture {
  x86_64,
  intel_gpu_bdw,
  intel_gpu_skl,
  intel_gpu_kbl,
  intel_gpu_cfl,
  intel_gpu_apl,
  intel_gpu_glk,
  intel_gpu_whl,
  intel_gpu_aml,
  intel_gpu_cml,
  intel_gpu_icllp,
  intel_gpu_ehl,
  intel_gpu_tgllp,
  intel_gpu_rkl,
  intel_gpu_adl_s,
  intel_gpu_rpl_s,
  intel_gpu_adl_p,
  intel_gpu_adl_n,
  intel_gpu_dg1,
  intel_gpu_acm_g10,
  intel_gpu_acm_g11,
  intel_gpu_acm_g12,
  intel_gpu_pvc,
  // Update "detail::max_architecture" below if you add new elements here!
  intel_gpu_8_0_0 = intel_gpu_bdw,
  intel_gpu_9_0_9 = intel_gpu_skl,
  intel_gpu_9_1_9 = intel_gpu_kbl,
  intel_gpu_9_2_9 = intel_gpu_cfl,
  intel_gpu_9_3_0 = intel_gpu_apl,
  intel_gpu_9_4_0 = intel_gpu_glk,
  intel_gpu_9_5_0 = intel_gpu_whl,
  intel_gpu_9_6_0 = intel_gpu_aml,
  intel_gpu_9_7_0 = intel_gpu_cml,
  intel_gpu_11_0_0 = intel_gpu_icllp,
  intel_gpu_11_2_0 = intel_gpu_ehl,
  intel_gpu_12_0_0 = intel_gpu_tgllp,
  intel_gpu_12_10_0 = intel_gpu_dg1,
};

} // namespace ext::intel::experimental

namespace detail {

static constexpr ext::intel::experimental::architecture max_architecture =
    ext::intel::experimental::architecture::intel_gpu_pvc;

#ifndef __SYCL_TARGET_INTEL_X86_64__
#define __SYCL_TARGET_INTEL_X86_64__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_BDW__
#define __SYCL_TARGET_INTEL_GPU_BDW__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_SKL__
#define __SYCL_TARGET_INTEL_GPU_SKL__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_KBL__
#define __SYCL_TARGET_INTEL_GPU_KBL__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_CFL__
#define __SYCL_TARGET_INTEL_GPU_CFL__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_APL__
#define __SYCL_TARGET_INTEL_GPU_APL__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_GLK__
#define __SYCL_TARGET_INTEL_GPU_GLK__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_WHL__
#define __SYCL_TARGET_INTEL_GPU_WHL__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_AML__
#define __SYCL_TARGET_INTEL_GPU_AML__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_CML__
#define __SYCL_TARGET_INTEL_GPU_CML__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_ICLLP__
#define __SYCL_TARGET_INTEL_GPU_ICLLP__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_EHL__
#define __SYCL_TARGET_INTEL_GPU_EHL__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_TGLLP__
#define __SYCL_TARGET_INTEL_GPU_TGLLP__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_RKL__
#define __SYCL_TARGET_INTEL_GPU_RKL__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_ADL_S__
#define __SYCL_TARGET_INTEL_GPU_ADL_S__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_RPL_S__
#define __SYCL_TARGET_INTEL_GPU_RPL_S__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_ADL_P__
#define __SYCL_TARGET_INTEL_GPU_ADL_P__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_ADL_N__
#define __SYCL_TARGET_INTEL_GPU_ADL_N__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_DG1__
#define __SYCL_TARGET_INTEL_GPU_DG1__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_ACM_G10__
#define __SYCL_TARGET_INTEL_GPU_ACM_G10__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_ACM_G11__
#define __SYCL_TARGET_INTEL_GPU_ACM_G11__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_ACM_G12__
#define __SYCL_TARGET_INTEL_GPU_ACM_G12__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_PVC__
#define __SYCL_TARGET_INTEL_GPU_PVC__ 0
#endif

// This is true when the translation unit is compiled in AOT mode with target
// names that supports the "if_architecture_is" features.  If an unsupported
// target name is specified via "-fsycl-targets", the associated invocation of
// the device compiler will set this variable to false, and that will trigger
// an error for code that uses "if_architecture_is".
static constexpr bool is_allowable_aot_mode =
    (__SYCL_TARGET_INTEL_X86_64__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_BDW__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_SKL__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_KBL__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_CFL__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_APL__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_GLK__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_WHL__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_AML__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_CML__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_ICLLP__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_EHL__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_TGLLP__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_RKL__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_ADL_S__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_RPL_S__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_ADL_P__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_ADL_N__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_DG1__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_ACM_G10__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_ACM_G11__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_ACM_G12__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_PVC__ == 1);

struct IsAOTForArchitectureClass {
  // Allocate an array of size == size of ext::intel::experimental::architecture
  // enum.
  bool arr[static_cast<int>(max_architecture) + 1];

  using arch = ext::intel::experimental::architecture;

  constexpr IsAOTForArchitectureClass() : arr() {
    arr[static_cast<int>(arch::x86_64)] = __SYCL_TARGET_INTEL_X86_64__ == 1;
    arr[static_cast<int>(arch::intel_gpu_bdw)] =
        __SYCL_TARGET_INTEL_GPU_BDW__ == 1;
    arr[static_cast<int>(arch::intel_gpu_skl)] =
        __SYCL_TARGET_INTEL_GPU_SKL__ == 1;
    arr[static_cast<int>(arch::intel_gpu_kbl)] =
        __SYCL_TARGET_INTEL_GPU_KBL__ == 1;
    arr[static_cast<int>(arch::intel_gpu_cfl)] =
        __SYCL_TARGET_INTEL_GPU_CFL__ == 1;
    arr[static_cast<int>(arch::intel_gpu_apl)] =
        __SYCL_TARGET_INTEL_GPU_APL__ == 1;
    arr[static_cast<int>(arch::intel_gpu_glk)] =
        __SYCL_TARGET_INTEL_GPU_GLK__ == 1;
    arr[static_cast<int>(arch::intel_gpu_whl)] =
        __SYCL_TARGET_INTEL_GPU_WHL__ == 1;
    arr[static_cast<int>(arch::intel_gpu_aml)] =
        __SYCL_TARGET_INTEL_GPU_AML__ == 1;
    arr[static_cast<int>(arch::intel_gpu_cml)] =
        __SYCL_TARGET_INTEL_GPU_CML__ == 1;
    arr[static_cast<int>(arch::intel_gpu_icllp)] =
        __SYCL_TARGET_INTEL_GPU_ICLLP__ == 1;
    arr[static_cast<int>(arch::intel_gpu_ehl)] =
        __SYCL_TARGET_INTEL_GPU_EHL__ == 1;
    arr[static_cast<int>(arch::intel_gpu_tgllp)] =
        __SYCL_TARGET_INTEL_GPU_TGLLP__ == 1;
    arr[static_cast<int>(arch::intel_gpu_rkl)] =
        __SYCL_TARGET_INTEL_GPU_RKL__ == 1;
    arr[static_cast<int>(arch::intel_gpu_adl_s)] =
        __SYCL_TARGET_INTEL_GPU_ADL_S__ == 1;
    arr[static_cast<int>(arch::intel_gpu_rpl_s)] =
        __SYCL_TARGET_INTEL_GPU_RPL_S__ == 1;
    arr[static_cast<int>(arch::intel_gpu_adl_p)] =
        __SYCL_TARGET_INTEL_GPU_ADL_P__ == 1;
    arr[static_cast<int>(arch::intel_gpu_adl_n)] =
        __SYCL_TARGET_INTEL_GPU_ADL_N__ == 1;
    arr[static_cast<int>(arch::intel_gpu_dg1)] =
        __SYCL_TARGET_INTEL_GPU_DG1__ == 1;
    arr[static_cast<int>(arch::intel_gpu_acm_g10)] =
        __SYCL_TARGET_INTEL_GPU_ACM_G10__ == 1;
    arr[static_cast<int>(arch::intel_gpu_acm_g11)] =
        __SYCL_TARGET_INTEL_GPU_ACM_G11__ == 1;
    arr[static_cast<int>(arch::intel_gpu_acm_g12)] =
        __SYCL_TARGET_INTEL_GPU_ACM_G12__ == 1;
    arr[static_cast<int>(arch::intel_gpu_pvc)] =
        __SYCL_TARGET_INTEL_GPU_PVC__ == 1;
  }
};

// One entry for each enumerator in "architecture" telling whether the AOT
// target matches that architecture.
static constexpr IsAOTForArchitectureClass is_aot_for_architecture;

// Reads the value of "is_allowable_aot_mode" via a template to defer triggering
// static_assert() until template instantiation time.
template <ext::intel::experimental::architecture... Archs>
constexpr static bool allowable_aot_mode() {
  return is_allowable_aot_mode;
}

// Tells if the current device has one of the architectures in the parameter
// pack.
template <ext::intel::experimental::architecture... Archs>
constexpr static bool device_architecture_is() {
  return (is_aot_for_architecture.arr[static_cast<int>(Archs)] || ...);
}

// Helper object used to implement "else_if_architecture_is" and "otherwise".
// The "MakeCall" template parameter tells whether a previous clause in the
// "if-elseif-elseif ..." chain was true.  When "MakeCall" is false, some
// previous clause was true, so none of the subsequent
// "else_if_architecture_is" or "otherwise" member functions should call the
// user's function.
template <bool MakeCall> class if_architecture_helper {
public:
  template <ext::intel::experimental::architecture... Archs, typename T,
            typename... Args>
  constexpr auto else_if_architecture_is(T fnTrue, Args... args) {
    if constexpr (MakeCall && device_architecture_is<Archs...>()) {
      fnTrue(args...);
      return if_architecture_helper<false>{};
    } else {
      (void)fnTrue;
      return if_architecture_helper<MakeCall>{};
    }
  }

  template <typename T, typename... Args>
  constexpr void otherwise(T fn, Args... args) {
    if constexpr (MakeCall) {
      fn(args...);
    }
  }
};
} // namespace detail

namespace ext::intel::experimental {

template <architecture... Archs, typename T, typename... Args>
constexpr static auto if_architecture_is(T fnTrue, Args... args) {
  static_assert(detail::allowable_aot_mode<Archs...>(),
                "The if_architecture_is function may only be used when AOT "
                "compiling with '-fsycl-targets=spir64_x86_64' or "
                "'-fsycl-targets=intel_gpu_*'");
  if constexpr (detail::device_architecture_is<Archs...>()) {
    fnTrue(args...);
    return detail::if_architecture_helper<false>{};
  } else {
    (void)fnTrue;
    return detail::if_architecture_helper<true>{};
  }
}

} // namespace ext::intel::experimental
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
