//==---------------- SyclLsArchitectureName.cpp ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include <sycl/ext/oneapi/experimental/device_architecture.hpp>

#include <string>
#include <vector>

// sycl-ls.cpp uses aspect::image; suppress the deprecation warning.
#define SYCL_DISABLE_IMAGE_ASPECT_WARNING
#define main sycl_ls_main
#include "../../tools/sycl-ls/sycl-ls.cpp"
#undef main

namespace syclex = sycl::ext::oneapi::experimental;

struct ArchitectureNameCase {
  syclex::architecture Arch;
  std::string Expected;
};

std::vector<ArchitectureNameCase> getArchitectureNameCases() {
  return {
      {syclex::architecture::intel_gpu_apl, "intel_gpu_apl / intel_gpu_bxt"},
      {syclex::architecture::intel_gpu_bxt, "intel_gpu_apl / intel_gpu_bxt"},
      {syclex::architecture::intel_gpu_icllp,
       "intel_gpu_icllp / intel_gpu_icl"},
      {syclex::architecture::intel_gpu_icl,
       "intel_gpu_icllp / intel_gpu_icl"},
      {syclex::architecture::intel_gpu_ehl, "intel_gpu_ehl / intel_gpu_jsl"},
      {syclex::architecture::intel_gpu_jsl, "intel_gpu_ehl / intel_gpu_jsl"},
      {syclex::architecture::intel_gpu_tgllp,
       "intel_gpu_tgllp / intel_gpu_tgl"},
      {syclex::architecture::intel_gpu_tgl,
       "intel_gpu_tgllp / intel_gpu_tgl"},
      {syclex::architecture::intel_gpu_adl_s,
       "intel_gpu_adl_s / intel_gpu_rpl_s"},
      {syclex::architecture::intel_gpu_rpl_s,
       "intel_gpu_adl_s / intel_gpu_rpl_s"},
      {syclex::architecture::intel_gpu_acm_g10,
       "intel_gpu_acm_g10 / intel_gpu_dg2_g10"},
      {syclex::architecture::intel_gpu_dg2_g10,
       "intel_gpu_acm_g10 / intel_gpu_dg2_g10"},
      {syclex::architecture::intel_gpu_acm_g11,
       "intel_gpu_acm_g11 / intel_gpu_dg2_g11"},
      {syclex::architecture::intel_gpu_dg2_g11,
       "intel_gpu_acm_g11 / intel_gpu_dg2_g11"},
      {syclex::architecture::intel_gpu_acm_g12,
       "intel_gpu_acm_g12 / intel_gpu_dg2_g12"},
      {syclex::architecture::intel_gpu_dg2_g12,
       "intel_gpu_acm_g12 / intel_gpu_dg2_g12"},
      {syclex::architecture::intel_gpu_mtl_u,
       "intel_gpu_mtl_u / intel_gpu_mtl_s / intel_gpu_arl_u / "
       "intel_gpu_arl_s"},
      {syclex::architecture::intel_gpu_mtl_s,
       "intel_gpu_mtl_u / intel_gpu_mtl_s / intel_gpu_arl_u / "
       "intel_gpu_arl_s"},
      {syclex::architecture::intel_gpu_arl_u,
       "intel_gpu_mtl_u / intel_gpu_mtl_s / intel_gpu_arl_u / "
       "intel_gpu_arl_s"},
      {syclex::architecture::intel_gpu_arl_s,
       "intel_gpu_mtl_u / intel_gpu_mtl_s / intel_gpu_arl_u / "
       "intel_gpu_arl_s"},
      {syclex::architecture::intel_gpu_nvl_s,
       "intel_gpu_nvl_s / intel_gpu_nvl_hx / intel_gpu_nvl_ul"},
      {syclex::architecture::intel_gpu_nvl_hx,
       "intel_gpu_nvl_s / intel_gpu_nvl_hx / intel_gpu_nvl_ul"},
      {syclex::architecture::intel_gpu_nvl_ul,
       "intel_gpu_nvl_s / intel_gpu_nvl_hx / intel_gpu_nvl_ul"},
      {syclex::architecture::intel_gpu_nvl_u,
       "intel_gpu_nvl_u / intel_gpu_nvl_h"},
      {syclex::architecture::intel_gpu_nvl_h,
       "intel_gpu_nvl_u / intel_gpu_nvl_h"},
  };
}

TEST(SyclLsArchitectureNameTest, PrintsAllAliases) {
  for (const auto &Case : getArchitectureNameCases())
    EXPECT_EQ(getArchitectureName(Case.Arch), Case.Expected);
}

TEST(SyclLsArchitectureNameTest, HidesNumericAliasesFromOutput) {
  EXPECT_EQ(getArchitectureName(syclex::architecture::intel_gpu_mtl_u),
            "intel_gpu_mtl_u / intel_gpu_mtl_s / intel_gpu_arl_u / "
            "intel_gpu_arl_s");
  EXPECT_EQ(getArchitectureName(syclex::architecture::intel_gpu_bmg_g21),
            "intel_gpu_bmg_g21");
}