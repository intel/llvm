// RUN: %clangxx -fsycl -fsyntax-only %s

#include <sycl/ext/oneapi/experimental/chunk.hpp>
#include <sycl/ext/oneapi/experimental/fragment.hpp>
#include <sycl/ext/oneapi/experimental/tangle.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

// check each trait correctly identifies own type
static_assert(syclex::is_chunk_v<syclex::chunk<8, sycl::sub_group>>);
static_assert(syclex::is_fragment_v<syclex::fragment<sycl::sub_group>>);
static_assert(syclex::is_tangle_v<syclex::tangle<sycl::sub_group>>);

// check traits return false for different group types (cross-check)
static_assert(!syclex::is_chunk_v<syclex::fragment<sycl::sub_group>>);
static_assert(!syclex::is_chunk_v<syclex::tangle<sycl::sub_group>>);
static_assert(!syclex::is_fragment_v<syclex::chunk<8, sycl::sub_group>>);
static_assert(!syclex::is_fragment_v<syclex::tangle<sycl::sub_group>>);
static_assert(!syclex::is_tangle_v<syclex::chunk<8, sycl::sub_group>>);
static_assert(!syclex::is_tangle_v<syclex::fragment<sycl::sub_group>>);

// check traits return false for base group types
static_assert(!syclex::is_chunk_v<sycl::sub_group>);
static_assert(!syclex::is_fragment_v<sycl::sub_group>);
static_assert(!syclex::is_tangle_v<sycl::sub_group>);

// chunk sizes
static_assert(syclex::is_chunk_v<syclex::chunk<1, sycl::sub_group>>);
static_assert(syclex::is_chunk_v<syclex::chunk<32, sycl::sub_group>>);

// these traits are used in spirv.hpp MapShuffleID() for dispatch
