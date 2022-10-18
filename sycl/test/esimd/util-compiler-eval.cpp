// RUN: %clangxx -fsyntax-only -fsycl %s

// This test checks that several utility APIs can be successfully compiled.

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::intel::experimental::esimd;

enum class TheEnum { FirstMember, SecondMember, ThirdMember };

static_assert(detail::is_one_of_v<float, float>, "");
static_assert(!detail::is_one_of_v<float, double>, "");
static_assert(
    detail::is_one_of_enum_v<TheEnum, TheEnum::FirstMember,
                             TheEnum::SecondMember, TheEnum::FirstMember>,
    "");
static_assert(
    !detail::is_one_of_enum_v<TheEnum, TheEnum::FirstMember,
                              TheEnum::SecondMember, TheEnum::ThirdMember>,
    "");
#ifndef __SYCL_DEVICE_ONLY__
static_assert(__ESIMD_EMU_DNS::is_hf_type<sycl::half>::value,
    "");
static_assert(!__ESIMD_EMU_DNS::is_hf_type<float>::value, "");
#endif
