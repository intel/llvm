// This is auto-generated SYCL integration header.

#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/kernel_desc.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/access/access.hpp>

#ifndef SYCL_LANGUAGE_VERSION
#define SYCL_LANGUAGE_VERSION 202012L
#endif //SYCL_LANGUAGE_VERSION

// Forward declarations of templated kernel function types:

namespace sycl {
inline namespace _V1 {
namespace detail {
// names of all kernels defined in the corresponding source
static constexpr
const char* const kernel_names[] = {
  "_Z19__sycl_kernel_emptyv",
  "_Z24__sycl_kernel_initializePi",
  "_Z23__sycl_kernel_successorPiS_",
  "_Z20__sycl_kernel_squarePiS_",
  "_Z22__sycl_kernel_square2DPiS_",
  "_Z22__sycl_kernel_square3DPiS_",
  "_Z32__sycl_kernel_squareWithAccessorN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE1ENS0_3ext6oneapi22accessor_property_listIJEEEEESA_",
  "_Z46__sycl_kernel_squareWithScratchMemoryTemplatedIiEvPT_S1_",
  "",
};

static constexpr unsigned kernel_args_sizes[] = {0, 1, 2, 2, 2, 2, 2, 2, 4294967295, 
};

// array representing signatures of all kernels defined in the
// corresponding source
static constexpr
const kernel_param_desc_t kernel_signatures[] = {
  //--- _Z19__sycl_kernel_emptyv

  //--- _Z24__sycl_kernel_initializePi
  { kernel_param_kind_t::kind_pointer, 8, 0 },

  //--- _Z23__sycl_kernel_successorPiS_
  { kernel_param_kind_t::kind_pointer, 8, 0 },
  { kernel_param_kind_t::kind_pointer, 8, 0 },

  //--- _Z20__sycl_kernel_squarePiS_
  { kernel_param_kind_t::kind_pointer, 8, 0 },
  { kernel_param_kind_t::kind_pointer, 8, 0 },

  //--- _Z22__sycl_kernel_square2DPiS_
  { kernel_param_kind_t::kind_pointer, 8, 0 },
  { kernel_param_kind_t::kind_pointer, 8, 0 },

  //--- _Z22__sycl_kernel_square3DPiS_
  { kernel_param_kind_t::kind_pointer, 8, 0 },
  { kernel_param_kind_t::kind_pointer, 8, 0 },

  //--- _Z32__sycl_kernel_squareWithAccessorN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE1ENS0_3ext6oneapi22accessor_property_listIJEEEEESA_
  { kernel_param_kind_t::kind_accessor, 4062, 0 },
  { kernel_param_kind_t::kind_accessor, 4062, 0 },

  //--- _Z46__sycl_kernel_squareWithScratchMemoryTemplatedIiEvPT_S1_
  { kernel_param_kind_t::kind_pointer, 8, 0 },
  { kernel_param_kind_t::kind_pointer, 8, 0 },

  { kernel_param_kind_t::kind_invalid, -987654321, -987654321 }, 
};

} // namespace detail
} // namespace _V1
} // namespace sycl

// Definition of _Z19__sycl_kernel_emptyv as a free function kernel

// Forward declarations of kernel and its argument types:

void empty();
static constexpr auto __sycl_shim1() {
  return (void (*)())empty;
}

namespace sycl {
inline namespace _V1 {
namespace detail {
//Free Function Kernel info specialization for shim1
template <> struct FreeFunctionInfoData<__sycl_shim1()> {
  __SYCL_DLL_LOCAL
  static constexpr unsigned getNumParams() { return 0; }
  __SYCL_DLL_LOCAL
  static constexpr const char *getFunctionName() { return "_Z19__sycl_kernel_emptyv"; }
};
} // namespace detail
} // namespace _V1
} // namespace sycl

namespace sycl {
template <>
struct ext::oneapi::experimental::is_kernel<__sycl_shim1()> {
  static constexpr bool value = true;
};
template <>
struct ext::oneapi::experimental::is_single_task_kernel<__sycl_shim1()> {
  static constexpr bool value = true;
};
}

// Definition of _Z24__sycl_kernel_initializePi as a free function kernel

// Forward declarations of kernel and its argument types:

void initialize(int * ptr);
static constexpr auto __sycl_shim2() {
  return (void (*)(int *))initialize;
}

namespace sycl {
inline namespace _V1 {
namespace detail {
//Free Function Kernel info specialization for shim2
template <> struct FreeFunctionInfoData<__sycl_shim2()> {
  __SYCL_DLL_LOCAL
  static constexpr unsigned getNumParams() { return 1; }
  __SYCL_DLL_LOCAL
  static constexpr const char *getFunctionName() { return "_Z24__sycl_kernel_initializePi"; }
};
} // namespace detail
} // namespace _V1
} // namespace sycl

namespace sycl {
template <>
struct ext::oneapi::experimental::is_kernel<__sycl_shim2()> {
  static constexpr bool value = true;
};
template <>
struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim2(), 1> {
  static constexpr bool value = true;
};
}

// Definition of _Z23__sycl_kernel_successorPiS_ as a free function kernel

// Forward declarations of kernel and its argument types:

void successor(int * src, int * dst);
static constexpr auto __sycl_shim3() {
  return (void (*)(int *, int *))successor;
}

namespace sycl {
inline namespace _V1 {
namespace detail {
//Free Function Kernel info specialization for shim3
template <> struct FreeFunctionInfoData<__sycl_shim3()> {
  __SYCL_DLL_LOCAL
  static constexpr unsigned getNumParams() { return 2; }
  __SYCL_DLL_LOCAL
  static constexpr const char *getFunctionName() { return "_Z23__sycl_kernel_successorPiS_"; }
};
} // namespace detail
} // namespace _V1
} // namespace sycl

namespace sycl {
template <>
struct ext::oneapi::experimental::is_kernel<__sycl_shim3()> {
  static constexpr bool value = true;
};
template <>
struct ext::oneapi::experimental::is_single_task_kernel<__sycl_shim3()> {
  static constexpr bool value = true;
};
}

// Definition of _Z20__sycl_kernel_squarePiS_ as a free function kernel

// Forward declarations of kernel and its argument types:

void square(int * src, int * dst);
static constexpr auto __sycl_shim4() {
  return (void (*)(int *, int *))square;
}

namespace sycl {
inline namespace _V1 {
namespace detail {
//Free Function Kernel info specialization for shim4
template <> struct FreeFunctionInfoData<__sycl_shim4()> {
  __SYCL_DLL_LOCAL
  static constexpr unsigned getNumParams() { return 2; }
  __SYCL_DLL_LOCAL
  static constexpr const char *getFunctionName() { return "_Z20__sycl_kernel_squarePiS_"; }
};
} // namespace detail
} // namespace _V1
} // namespace sycl

namespace sycl {
template <>
struct ext::oneapi::experimental::is_kernel<__sycl_shim4()> {
  static constexpr bool value = true;
};
template <>
struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim4(), 1> {
  static constexpr bool value = true;
};
}

// Definition of _Z22__sycl_kernel_square2DPiS_ as a free function kernel

// Forward declarations of kernel and its argument types:

void square2D(int * src, int * dst);
static constexpr auto __sycl_shim5() {
  return (void (*)(int *, int *))square2D;
}

namespace sycl {
inline namespace _V1 {
namespace detail {
//Free Function Kernel info specialization for shim5
template <> struct FreeFunctionInfoData<__sycl_shim5()> {
  __SYCL_DLL_LOCAL
  static constexpr unsigned getNumParams() { return 2; }
  __SYCL_DLL_LOCAL
  static constexpr const char *getFunctionName() { return "_Z22__sycl_kernel_square2DPiS_"; }
};
} // namespace detail
} // namespace _V1
} // namespace sycl

namespace sycl {
template <>
struct ext::oneapi::experimental::is_kernel<__sycl_shim5()> {
  static constexpr bool value = true;
};
template <>
struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim5(), 2> {
  static constexpr bool value = true;
};
}

// Definition of _Z22__sycl_kernel_square3DPiS_ as a free function kernel

// Forward declarations of kernel and its argument types:

void square3D(int * src, int * dst);
static constexpr auto __sycl_shim6() {
  return (void (*)(int *, int *))square3D;
}

namespace sycl {
inline namespace _V1 {
namespace detail {
//Free Function Kernel info specialization for shim6
template <> struct FreeFunctionInfoData<__sycl_shim6()> {
  __SYCL_DLL_LOCAL
  static constexpr unsigned getNumParams() { return 2; }
  __SYCL_DLL_LOCAL
  static constexpr const char *getFunctionName() { return "_Z22__sycl_kernel_square3DPiS_"; }
};
} // namespace detail
} // namespace _V1
} // namespace sycl

namespace sycl {
template <>
struct ext::oneapi::experimental::is_kernel<__sycl_shim6()> {
  static constexpr bool value = true;
};
template <>
struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim6(), 3> {
  static constexpr bool value = true;
};
}

// Definition of _Z32__sycl_kernel_squareWithAccessorN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE1ENS0_3ext6oneapi22accessor_property_listIJEEEEESA_ as a free function kernel

// Forward declarations of kernel and its argument types:
namespace sycl { inline namespace _V1 { namespace access { 
enum class mode : int;
}}}
namespace sycl { inline namespace _V1 { namespace access { 
enum class target : int;
}}}
namespace sycl { inline namespace _V1 { namespace access { 
enum class placeholder : int;
}}}
namespace sycl { inline namespace _V1 { namespace ext { namespace oneapi { 
template <typename ...PropsT> class accessor_property_list;
}}}}
namespace sycl { inline namespace _V1 { 
template <typename DataT, int Dimensions, sycl::access::mode AccessMode, sycl::access::target AccessTarget, sycl::access::placeholder IsPlaceholder, typename PropertyListT> class accessor;
}}

void squareWithAccessor(sycl::accessor<int, 1, static_cast<sycl::access::mode>(1026), static_cast<sycl::access::target>(2014), static_cast<sycl::access::placeholder>(1), sycl::ext::oneapi::accessor_property_list<>> src, sycl::accessor<int, 1, static_cast<sycl::access::mode>(1026), static_cast<sycl::access::target>(2014), static_cast<sycl::access::placeholder>(1), sycl::ext::oneapi::accessor_property_list<>> dst);
static constexpr auto __sycl_shim7() {
  return (void (*)(class sycl::accessor<int, 1, static_cast<sycl::access::mode>(1026), static_cast<sycl::access::target>(2014), static_cast<sycl::access::placeholder>(1), class sycl::ext::oneapi::accessor_property_list<>>, class sycl::accessor<int, 1, static_cast<sycl::access::mode>(1026), static_cast<sycl::access::target>(2014), static_cast<sycl::access::placeholder>(1), class sycl::ext::oneapi::accessor_property_list<>>))squareWithAccessor;
}

namespace sycl {
inline namespace _V1 {
namespace detail {
//Free Function Kernel info specialization for shim7
template <> struct FreeFunctionInfoData<__sycl_shim7()> {
  __SYCL_DLL_LOCAL
  static constexpr unsigned getNumParams() { return 2; }
  __SYCL_DLL_LOCAL
  static constexpr const char *getFunctionName() { return "_Z32__sycl_kernel_squareWithAccessorN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE1ENS0_3ext6oneapi22accessor_property_listIJEEEEESA_"; }
};
} // namespace detail
} // namespace _V1
} // namespace sycl

namespace sycl {
template <>
struct ext::oneapi::experimental::is_kernel<__sycl_shim7()> {
  static constexpr bool value = true;
};
template <>
struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim7(), 1> {
  static constexpr bool value = true;
};
}

// Definition of _Z46__sycl_kernel_squareWithScratchMemoryTemplatedIiEvPT_S1_ as a free function kernel

// Forward declarations of kernel and its argument types:

template <typename T> void squareWithScratchMemoryTemplated(T *, T *);
static constexpr auto __sycl_shim8() {
  return (void (*)(int *, int *))squareWithScratchMemoryTemplated<int>;
}

namespace sycl {
inline namespace _V1 {
namespace detail {
//Free Function Kernel info specialization for shim8
template <> struct FreeFunctionInfoData<__sycl_shim8()> {
  __SYCL_DLL_LOCAL
  static constexpr unsigned getNumParams() { return 2; }
  __SYCL_DLL_LOCAL
  static constexpr const char *getFunctionName() { return "_Z46__sycl_kernel_squareWithScratchMemoryTemplatedIiEvPT_S1_"; }
};
} // namespace detail
} // namespace _V1
} // namespace sycl

namespace sycl {
template <>
struct ext::oneapi::experimental::is_kernel<__sycl_shim8()> {
  static constexpr bool value = true;
};
template <>
struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim8(), 1> {
  static constexpr bool value = true;
};
}

#include <sycl/kernel_bundle.hpp>
#include <sycl/detail/kernel_global_info.hpp>
namespace {
struct GlobalMapUpdater {
  GlobalMapUpdater() {
    sycl::detail::free_function_info_map::add(sycl::detail::kernel_names, sycl::detail::kernel_args_sizes, 8);
  }
  ~GlobalMapUpdater() {
    sycl::detail::free_function_info_map::remove(sycl::detail::kernel_names, sycl::detail::kernel_args_sizes, 8);
  }
};
static GlobalMapUpdater updater;
} // namespace
// Specializations of KernelInfo for kernel function types:
namespace sycl {
inline namespace _V1 {
namespace detail {

} // namespace detail
} // namespace _V1
} // namespace sycl
