//===-- ctor_load_acc_coverage.hpp - Define coverage for ctor_load_acc tests
//      -------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// As we generally need to duplicate coverage for core and fp_extra types, it's
/// better to have some common place to store type and value packs definition
///
//===----------------------------------------------------------------------===//

#pragma once

#include "../sycl_accessor.hpp"
#include "ctor_load_acc.hpp"

namespace esimd_test::api::functional::ctors::coverage {

//----------------------------------------------------------------------------//
// Define input for sycl::accessor<T,Dims,Mode,Target> type combinations
//----------------------------------------------------------------------------//

inline auto get_all_dimensions() {
#ifdef ESIMD_TESTS_FULL_COVERAGE
  return get_accessor_dimensions();
#else
  // Verify both the single-dimension and multi-dimension access
  return integer_pack<1, 3>::generate_unnamed();
#endif
}

inline auto get_all_modes() {
#ifdef ESIMD_TESTS_FULL_COVERAGE
  return accessor_modes<accessor_mode_group::all_read>::generate_named();
#else
  // Use any single read mode from the possible ones
  return accessor_modes<accessor_mode_group::single_read>::generate_named();
#endif
}

inline auto get_all_targets() { return accessor_targets::generate_named(); }

//----------------------------------------------------------------------------//
// Define input for constructor calls
//----------------------------------------------------------------------------//

inline auto get_all_contexts() {
  return unnamed_type_pack<ctors::initializer, ctors::var_decl,
                           ctors::rval_in_expr, ctors::const_ref>::generate();
}

inline auto get_all_offset_generators() {
#ifdef ESIMD_TESTS_FULL_COVERAGE
  return unnamed_type_pack<offset_generator<0>, offset_generator<1>,
                           offset_generator<2>>::generate();
#else
  // Verify all cases with both zero and non-zero offsets
  return unnamed_type_pack<offset_generator<0>,
                           offset_generator<1>>::generate();
#endif
}
} // namespace esimd_test::api::functional::ctors::coverage
