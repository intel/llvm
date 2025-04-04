#pragma once

#include <sycl/detail/defines_elementary.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

#ifndef __SYCL_DEVICE_ONLY__
#define _Bool bool
#endif

  // kernel parameter kinds
  enum class kernel_param_kind_t {
    kind_accessor = 0,
    kind_std_layout = 1, // standard layout object parameters
    kind_sampler = 2,
    kind_pointer = 3,
    kind_specialization_constants_buffer = 4,
    kind_stream = 5,
    kind_work_group_memory = 6,
    kind_dynamic_work_group_memory = 7,
    kind_invalid = 0xf, // not a valid kernel kind
  };

  // describes a kernel parameter
  struct kernel_param_desc_t {
    // parameter kind
    kernel_param_kind_t kind;
    // kind == kind_std_layout
    //   parameter size in bytes (includes padding for structs)
    // kind == kind_accessor
    //   access target; possible access targets are defined in access/access.hpp
    int info;
    // offset of the captured value of the parameter in the lambda or function
    // object
    int offset;
  };

  template <bool Cond, typename TrueT, typename FalseT>
  struct conditional {
    using type = TrueT;
  };
  template <typename TrueT, typename FalseT>
  struct conditional<false, TrueT, FalseT> {
    using type = FalseT;
  };

  using int64_t = conditional<sizeof(long) == 8, long, long long>::type;

  template <class KernelNameType> struct KernelInfo {
    static constexpr unsigned getNumParams() { return 0; }
    static const kernel_param_desc_t &getParamDesc(int) {
      static kernel_param_desc_t Dummy;
      return Dummy;
    }
    static constexpr const char *getName() { return ""; }
    static constexpr bool isESIMD() { return 0; }
    static constexpr int64_t getKernelSize() { return 0; }
  };
} // namespace detail
} // namespace _V1
} // namespace sycl
