//==-------------- alt_ui.hpp - DPC++ Explicit SIMD API   ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// "Alternative" convenience Explicit SIMD APIs.
//===----------------------------------------------------------------------===//

#include <sycl/ext/intel/esimd/simd.hpp>
#include <sycl/ext/intel/esimd/simd_view.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace __ESIMD_NS {

/// @addtogroup sycl_esimd_misc
/// @{

/// "Merges" elements of the input simd object according to the merge mask.
/// @param a The first simd object.
/// @param b The second simd object.
/// @param m The merge mask.
/// @return A simd object, where each element equals to corresponding element
///   from \c a if corresponding merge mask element is non-zero or element
///   from \c b otherwise.
template <class T, int N>
__ESIMD_API simd<T, N> merge(simd<T, N> a, simd<T, N> b, simd_mask<N> m) {
  b.merge(a, m);
  return b;
}

/// "Merges" elements of the input masks according to the merge mask.
/// @param a The first mask.
/// @param b The second mask.
/// @param m The merge mask.
/// @return A mask, where each element equals to corresponding element from
///    \c a if corresponding merge mask element is non-zero or \c element from
///    \c b otherwise.
template <int N>
__ESIMD_API simd_mask<N> merge(simd_mask<N> a, simd_mask<N> b, simd_mask<N> m) {
  b.merge(a, m);
  return a;
}

/// "Merges" elements of vectors referenced by the input views.
/// Available only when all of the length and the element type of the subregions
/// referenced by both input views are the same.
/// @param a The first view.
/// @param b The second view.
/// @param m The merge mask.
/// @return A vector (mask or simd object), where each element equals to
///   corresponding element from \c a if corresponding merge mask element is
///   non-zero or \c element from \c b otherwise.
template <class BaseT1, class BaseT2, class RegionT1, class RegionT2,
          class = std::enable_if_t<
              (shape_type<RegionT1>::length == shape_type<RegionT2>::length) &&
              std::is_same_v<detail::element_type_t<BaseT1>,
                             detail::element_type_t<BaseT2>>>>
__ESIMD_API auto merge(simd_view<BaseT1, RegionT1> v1,
                       simd_view<BaseT2, RegionT2> v2,
                       simd_mask<shape_type<RegionT1>::length> m) {
  return merge(v1.read(), v2.read(), m);
}

/// @} sycl_esimd_misc

} // namespace __ESIMD_NS
} // __SYCL_INLINE_NAMESPACE(cl)
