// RUN: %clangxx -fsycl -fsycl-device-only -c %s
// This test checks compile-time evaluation of functions from esimd_util.hpp

#include "CL/sycl.hpp"
#include "CL/sycl/INTEL/esimd/esimd.hpp"

static_assert(__esimd::getNextPowerOf2<0>() == 0, "");
static_assert(__esimd::getNextPowerOf2<1>() == 1, "");
static_assert(__esimd::getNextPowerOf2<7>() == 8, "");
static_assert(__esimd::getNextPowerOf2<1024>() == 1024, "");

static_assert(__esimd::log2<0>() == 0, "");
static_assert(__esimd::log2<1>() == 0, "");
static_assert(__esimd::log2<7>() == 2, "");
static_assert(__esimd::log2<1024 * 1024>() == 20, "");
