// DEFINE: %{common_opts_spirv32} = -internal-isystem %S/Inputs \
// DEFINE: -fsycl-is-device -emit-llvm -triple spirv32-unknown-unknown

// DEFINE: %{common_opts_spirv64} = -internal-isystem %S/Inputs \
// DEFINE: -fsycl-is-device -emit-llvm -triple spirv64-unknown-unknown

// DEFINE: %{common_opts_spir} = -internal-isystem %S/Inputs \
// DEFINE: -fsycl-is-device -emit-llvm -triple spir-unknown-unknown

// DEFINE: %{common_opts_spir64} = -internal-isystem %S/Inputs \
// DEFINE: -fsycl-is-device -emit-llvm -triple spir64-unknown-unknown

// RUN: %clang_cc1 %{common_opts_spirv32} %s -o - \
// RUN: | FileCheck --check-prefix PREC-SQRT %s

// RUN: %clang_cc1 %{common_opts_spirv32} -foffload-fp32-prec-sqrt %s -o - \
// RUN: | FileCheck --check-prefix PREC-SQRT %s

// RUN: %clang_cc1 %{common_opts_spirv32} -fno-offload-fp32-prec-sqrt %s -o - \
// RUN: | FileCheck --check-prefix ROUNDED-SQRT %s

// RUN: %clang_cc1 %{common_opts_spirv32} -foffload-fp32-prec-div %s -o - \
// RUN: | FileCheck --check-prefix PREC-DIV %s

// RUN: %clang_cc1 %{common_opts_spirv32} -fno-offload-fp32-prec-div %s -o - \
// RUN: | FileCheck --check-prefix ROUNDED-DIV %s

// RUN: %clang_cc1 %{common_opts_spirv32} -fno-offload-fp32-prec-div \
// RUN: -fno-offload-fp32-prec-sqrt %s -o - \
// RUN: | FileCheck --check-prefix ROUNDED-DIV-ROUNDED-SQRT %s

// RUN: %clang_cc1 %{common_opts_spirv32} -ffast-math \
// RUN:-fno-offload-fp32-prec-div -fno-offload-fp32-prec-sqrt %s -o - \
// RUN: | FileCheck --check-prefix ROUNDED-SQRT-FAST %s

// RUN: %clang_cc1 %{common_opts_spirv32} -ffast-math \
// RUN: -fno-offload-fp32-prec-div -fno-offload-fp32-prec-sqrt %s -o - \
// RUN: | FileCheck --check-prefix ROUNDED-DIV-FAST %s

// RUN: %clang_cc1 %{common_opts_spirv32} -ffast-math -foffload-fp32-prec-div \
// RUN: %s -o - | FileCheck --check-prefix PREC-FAST %s

// RUN: %clang_cc1 %{common_opts_spirv32} -ffast-math -foffload-fp32-prec-sqrt \
// RUN: %s -o - | FileCheck --check-prefix PREC-FAST %s

// RUN: %clang_cc1 %{common_opts_spirv32} -ffast-math \
// RUN: -fno-offload-fp32-prec-div -foffload-fp32-prec-sqrt %s -o - \
// RUN: | FileCheck --check-prefix PREC-SQRT-FAST %s

// RUN: %clang_cc1 %{common_opts_spirv32} -ffast-math \
// RUN: -fno-offload-fp32-prec-sqrt -foffload-fp32-prec-div \
// RUN: %s -o - | FileCheck --check-prefix ROUNDED-SQRT-PREC-DIV %s

// RUN: %clang_cc1 %{common_opts_spirv32} -ffast-math \
// RUN: -fno-offload-fp32-prec-div -foffload-fp32-prec-sqrt \
// RUN: %s -o - | FileCheck --check-prefix ROUNDED-DIV-PREC-SQRT %s

// RUN: %clang_cc1 %{common_opts_spirv32} -ffast-math \
// RUN: -fno-offload-fp32-prec-div -fno-offload-fp32-prec-sqrt \
// RUN: %s -o - | FileCheck --check-prefix ROUNDED-DIV-ROUNDED-SQRT-FAST %s

// RUN: %clang_cc1 %{common_opts_spirv32} -fno-offload-fp32-prec-div \
// RUN: -ffp-builtin-accuracy=high %s -o - \
// RUN: | FileCheck --check-prefix LOW-PREC-DIV %s

// RUN: %clang_cc1 %{common_opts_spirv32} -ffp-builtin-accuracy=high:div \
// RUN: -fno-offload-fp32-prec-div %s -o - \
// RUN: | FileCheck --check-prefix ROUNDED-DIV %s

// RUN: %clang_cc1 %{common_opts_spirv32} -fno-offload-fp32-prec-sqrt \
// RUN: -ffp-builtin-accuracy=high %s -o - \
// RUN: | FileCheck --check-prefix LOW-PREC-SQRT %s

// RUN: %clang_cc1 %{common_opts_spirv32} -fno-offload-fp32-prec-div \
// RUN: -ffp-builtin-accuracy=high:sin %s -o - \
// RUN: | FileCheck --check-prefix ROUNDED-DIV %s

// RUN: %clang_cc1 %{common_opts_spirv32} -fno-offload-fp32-prec-sqrt \
// RUN: -ffp-builtin-accuracy=high:sin %s -o - \
// RUN: | FileCheck --check-prefix ROUNDED-SQRT %s

//

// RUN: %clang_cc1 %{common_opts_spirv64} %s -o - \
// RUN: | FileCheck --check-prefix PREC-SQRT %s

// RUN: %clang_cc1 %{common_opts_spirv64} -foffload-fp32-prec-sqrt %s -o - \
// RUN: | FileCheck --check-prefix PREC-SQRT %s

// RUN: %clang_cc1 %{common_opts_spirv64} -fno-offload-fp32-prec-sqrt %s -o - \
// RUN: | FileCheck --check-prefix ROUNDED-SQRT %s

// RUN: %clang_cc1 %{common_opts_spirv64} -foffload-fp32-prec-div %s -o - \
// RUN: | FileCheck --check-prefix PREC-DIV %s

// RUN: %clang_cc1 %{common_opts_spirv64} -fno-offload-fp32-prec-div %s -o - \
// RUN: | FileCheck --check-prefix ROUNDED-DIV %s

// RUN: %clang_cc1 %{common_opts_spirv64} -fno-offload-fp32-prec-div \
// RUN: -fno-offload-fp32-prec-sqrt %s -o - \
// RUN: | FileCheck --check-prefix ROUNDED-DIV-ROUNDED-SQRT %s

// RUN: %clang_cc1 %{common_opts_spirv64} -ffast-math -fno-offload-fp32-prec-div \
// RUN: -fno-offload-fp32-prec-sqrt %s -o - \
// RUN: | FileCheck --check-prefix ROUNDED-SQRT-FAST %s

// RUN: %clang_cc1 %{common_opts_spirv64} -ffast-math -fno-offload-fp32-prec-div \
// RUN: -fno-offload-fp32-prec-sqrt %s -o - \
// RUN: | FileCheck --check-prefix ROUNDED-DIV-FAST %s

// RUN: %clang_cc1 %{common_opts_spirv64} -ffast-math -foffload-fp32-prec-div \
// RUN: %s -o - | FileCheck --check-prefix PREC-FAST %s

// RUN: %clang_cc1 %{common_opts_spirv64} -ffast-math \
// RUN: -fno-offload-fp32-prec-div -foffload-fp32-prec-sqrt %s -o - \
// RUN: | FileCheck --check-prefix PREC-SQRT-FAST %s

// RUN: %clang_cc1 %{common_opts_spirv64} -ffast-math \
// RUN: -fno-offload-fp32-prec-sqrt -foffload-fp32-prec-div \
// RUN: %s -o - | FileCheck --check-prefix ROUNDED-SQRT-PREC-DIV %s

// RUN: %clang_cc1 %{common_opts_spirv64} -ffast-math \
// RUN: -fno-offload-fp32-prec-div -foffload-fp32-prec-sqrt \
// RUN: %s -o - | FileCheck --check-prefix ROUNDED-DIV-PREC-SQRT %s

// RUN: %clang_cc1 %{common_opts_spirv64} -ffast-math \
// RUN: -fno-offload-fp32-prec-div -fno-offload-fp32-prec-sqrt \
// RUN: %s -o - | FileCheck --check-prefix ROUNDED-DIV-ROUNDED-SQRT-FAST %s

// RUN: %clang_cc1 %{common_opts_spirv64} -fno-offload-fp32-prec-div \
// RUN: -ffp-builtin-accuracy=high %s -o - \
// RUN: | FileCheck --check-prefix LOW-PREC-DIV %s

// RUN: %clang_cc1 %{common_opts_spirv64} -ffp-builtin-accuracy=high:div  \
// RUN: -fno-offload-fp32-prec-div %s -o - \
// RUN: | FileCheck --check-prefix ROUNDED-DIV %s

// RUN: %clang_cc1 %{common_opts_spirv64} -fno-offload-fp32-prec-sqrt \
// RUN: -ffp-builtin-accuracy=high %s -o - \
// RUN: | FileCheck --check-prefix LOW-PREC-SQRT %s

// RUN: %clang_cc1 %{common_opts_spirv64} -fno-offload-fp32-prec-div \
// RUN: -ffp-builtin-accuracy=high:sin %s -o - \
// RUN: | FileCheck --check-prefix ROUNDED-DIV %s

// RUN: %clang_cc1 %{common_opts_spirv64} -fno-offload-fp32-prec-sqrt \
// RUN: -ffp-builtin-accuracy=high:sin %s -o - \
// RUN: | FileCheck --check-prefix ROUNDED-SQRT %s

// RUN: %clang_cc1 %{common_opts_spir} %s -o - \
// RUN: | FileCheck --check-prefix PREC-SQRT %s

// RUN: %clang_cc1 %{common_opts_spir} -foffload-fp32-prec-sqrt %s -o - \
// RUN: | FileCheck --check-prefix PREC-SQRT %s

// RUN: %clang_cc1 %{common_opts_spir} -fno-offload-fp32-prec-sqrt %s -o - \
// RUN: | FileCheck --check-prefix ROUNDED-SQRT %s

// RUN: %clang_cc1 %{common_opts_spir} -foffload-fp32-prec-div %s -o - \
// RUN: | FileCheck --check-prefix PREC-DIV %s

// RUN: %clang_cc1 %{common_opts_spir} -fno-offload-fp32-prec-div %s -o - \
// RUN: | FileCheck --check-prefix ROUNDED-DIV %s

// RUN: %clang_cc1 %{common_opts_spir} -fno-offload-fp32-prec-div \
// RUN: -fno-offload-fp32-prec-sqrt %s -o - \
// RUN: | FileCheck --check-prefix ROUNDED-DIV-ROUNDED-SQRT %s

// RUN: %clang_cc1 %{common_opts_spir} -ffast-math \
// RUN:-fno-offload-fp32-prec-div -fno-offload-fp32-prec-sqrt %s -o - \
// RUN: | FileCheck --check-prefix ROUNDED-SQRT-FAST %s

// RUN: %clang_cc1 %{common_opts_spir} -ffast-math \
// RUN: -fno-offload-fp32-prec-div -fno-offload-fp32-prec-sqrt %s -o - \
// RUN: | FileCheck --check-prefix ROUNDED-DIV-FAST %s

// RUN: %clang_cc1 %{common_opts_spir} -ffast-math -foffload-fp32-prec-div \
// RUN: %s -o - | FileCheck --check-prefix PREC-FAST %s

// RUN: %clang_cc1 %{common_opts_spir} -ffast-math -foffload-fp32-prec-sqrt \
// RUN: %s -o - | FileCheck --check-prefix PREC-FAST %s

// RUN: %clang_cc1 %{common_opts_spir} -ffast-math \
// RUN: -fno-offload-fp32-prec-div -foffload-fp32-prec-sqrt %s -o - \
// RUN: | FileCheck --check-prefix PREC-SQRT-FAST %s

// RUN: %clang_cc1 %{common_opts_spir} -ffast-math \
// RUN: -fno-offload-fp32-prec-sqrt -foffload-fp32-prec-div \
// RUN: %s -o - | FileCheck --check-prefix ROUNDED-SQRT-PREC-DIV %s

// RUN: %clang_cc1 %{common_opts_spir} -ffast-math \
// RUN: -fno-offload-fp32-prec-div -foffload-fp32-prec-sqrt \
// RUN: %s -o - | FileCheck --check-prefix ROUNDED-DIV-PREC-SQRT %s

// RUN: %clang_cc1 %{common_opts_spir} -ffast-math \
// RUN: -fno-offload-fp32-prec-div -fno-offload-fp32-prec-sqrt \
// RUN: %s -o - | FileCheck --check-prefix ROUNDED-DIV-ROUNDED-SQRT-FAST %s

// RUN: %clang_cc1 %{common_opts_spir} -fno-offload-fp32-prec-div \
// RUN: -ffp-builtin-accuracy=high %s -o - \
// RUN: | FileCheck --check-prefix LOW-PREC-DIV %s

// RUN: %clang_cc1 %{common_opts_spir} -ffp-builtin-accuracy=high:div \
// RUN: -fno-offload-fp32-prec-div %s -o - \
// RUN: | FileCheck --check-prefix ROUNDED-DIV %s

// RUN: %clang_cc1 %{common_opts_spir} -fno-offload-fp32-prec-sqrt \
// RUN: -ffp-builtin-accuracy=high %s -o - \
// RUN: | FileCheck --check-prefix LOW-PREC-SQRT %s

// RUN: %clang_cc1 %{common_opts_spir} -fno-offload-fp32-prec-div \
// RUN: -ffp-builtin-accuracy=high:sin %s -o - \
// RUN: | FileCheck --check-prefix ROUNDED-DIV %s

// RUN: %clang_cc1 %{common_opts_spir} -fno-offload-fp32-prec-sqrt \
// RUN: -ffp-builtin-accuracy=high:sin %s -o - \
// RUN: | FileCheck --check-prefix ROUNDED-SQRT %s

//

// RUN: %clang_cc1 %{common_opts_spir64} %s -o - \
// RUN: | FileCheck --check-prefix PREC-SQRT %s

// RUN: %clang_cc1 %{common_opts_spir64} -foffload-fp32-prec-sqrt %s -o - \
// RUN: | FileCheck --check-prefix PREC-SQRT %s

// RUN: %clang_cc1 %{common_opts_spir64} -fno-offload-fp32-prec-sqrt %s -o - \
// RUN: | FileCheck --check-prefix ROUNDED-SQRT %s

// RUN: %clang_cc1 %{common_opts_spir64} -foffload-fp32-prec-div %s -o - \
// RUN: | FileCheck --check-prefix PREC-DIV %s

// RUN: %clang_cc1 %{common_opts_spir64} -fno-offload-fp32-prec-div %s -o - \
// RUN: | FileCheck --check-prefix ROUNDED-DIV %s

// RUN: %clang_cc1 %{common_opts_spir64} -fno-offload-fp32-prec-div \
// RUN: -fno-offload-fp32-prec-sqrt %s -o - \
// RUN: | FileCheck --check-prefix ROUNDED-DIV-ROUNDED-SQRT %s

// RUN: %clang_cc1 %{common_opts_spir64} -ffast-math -fno-offload-fp32-prec-div \
// RUN: -fno-offload-fp32-prec-sqrt %s -o - \
// RUN: | FileCheck --check-prefix ROUNDED-SQRT-FAST %s

// RUN: %clang_cc1 %{common_opts_spir64} -ffast-math -fno-offload-fp32-prec-div \
// RUN: -fno-offload-fp32-prec-sqrt %s -o - \
// RUN: | FileCheck --check-prefix ROUNDED-DIV-FAST %s

// RUN: %clang_cc1 %{common_opts_spir64} -ffast-math -foffload-fp32-prec-div \
// RUN: %s -o - | FileCheck --check-prefix PREC-FAST %s

// RUN: %clang_cc1 %{common_opts_spir64} -ffast-math \
// RUN: -fno-offload-fp32-prec-div -foffload-fp32-prec-sqrt %s -o - \
// RUN: | FileCheck --check-prefix PREC-SQRT-FAST %s

// RUN: %clang_cc1 %{common_opts_spir64} -ffast-math \
// RUN: -fno-offload-fp32-prec-sqrt -foffload-fp32-prec-div \
// RUN: %s -o - | FileCheck --check-prefix ROUNDED-SQRT-PREC-DIV %s

// RUN: %clang_cc1 %{common_opts_spir64} -ffast-math \
// RUN: -fno-offload-fp32-prec-div -foffload-fp32-prec-sqrt \
// RUN: %s -o - | FileCheck --check-prefix ROUNDED-DIV-PREC-SQRT %s

// RUN: %clang_cc1 %{common_opts_spir64} -ffast-math \
// RUN: -fno-offload-fp32-prec-div -fno-offload-fp32-prec-sqrt \
// RUN: %s -o - | FileCheck --check-prefix ROUNDED-DIV-ROUNDED-SQRT-FAST %s

// RUN: %clang_cc1 %{common_opts_spir64} -fno-offload-fp32-prec-div \
// RUN: -ffp-builtin-accuracy=high %s -o - \
// RUN: | FileCheck --check-prefix LOW-PREC-DIV %s

// RUN: %clang_cc1 %{common_opts_spir64} -ffp-builtin-accuracy=high:div  \
// RUN: -fno-offload-fp32-prec-div %s -o - \
// RUN: | FileCheck --check-prefix ROUNDED-DIV %s

// RUN: %clang_cc1 %{common_opts_spir64} -fno-offload-fp32-prec-sqrt \
// RUN: -ffp-builtin-accuracy=high %s -o - \
// RUN: | FileCheck --check-prefix LOW-PREC-SQRT %s

// RUN: %clang_cc1 %{common_opts_spir64} -fno-offload-fp32-prec-div \
// RUN: -ffp-builtin-accuracy=high:sin %s -o - \
// RUN: | FileCheck --check-prefix ROUNDED-DIV %s

// RUN: %clang_cc1 %{common_opts_spir64} -fno-offload-fp32-prec-sqrt \
// RUN: -ffp-builtin-accuracy=high:sin %s -o - \
// RUN: | FileCheck --check-prefix ROUNDED-SQRT %s

#include "sycl.hpp"

extern "C" SYCL_EXTERNAL float sqrt(float);

using namespace sycl;

int main() {
  const unsigned array_size = 4;
  range<1> numOfItems{array_size};
  float Value1 = .5f;
  float Value2 = .9f;
  queue deviceQueue;
  float *a;

  deviceQueue.submit([&](handler& cgh) {
    cgh.parallel_for<class KernelSqrt>(numOfItems,
    [=](id<1> wiID) {
      // PREC-SQRT: call spir_func float @sqrt(float noundef {{.*}})
      // ROUNDED-SQRT: call float @llvm.fpbuiltin.sqrt.f32(float {{.*}}) #[[ATTR_SQRT:[0-9]+]]
      // ROUNDED-SQRT-FAST: call reassoc nnan ninf nsz arcp afn float @llvm.fpbuiltin.sqrt.f32(float {{.*}}) #[[ATTR_SQRT:[0-9]+]]
      // PREC-DIV: call spir_func float @sqrt(float noundef {{.*}})
      // ROUNDED-DIV:  call spir_func float @sqrt(float noundef {{.*}})
      // ROUNDED-DIV-ROUNDED-SQRT: call float @llvm.fpbuiltin.sqrt.f32(float {{.*}}) #[[ATTR_SQRT:[0-9]+]]
      // PREC-FAST: call reassoc nnan ninf nsz arcp afn spir_func nofpclass(nan inf) float @sqrt(float noundef nofpclass(nan inf) {{.*}})
      // PREC-SQRT-FAST: call reassoc nnan ninf nsz arcp afn spir_func nofpclass(nan inf) float @sqrt(float noundef nofpclass(nan inf) {{.*}})
      // ROUNDED-SQRT-PREC-DIV: call reassoc nnan ninf nsz arcp afn float @llvm.fpbuiltin.sqrt.f32(float {{.*}}) #[[ATTR_SQRT:[0-9]+]]
      // ROUNDED-DIV-PREC-SQRT: call reassoc nnan ninf nsz arcp afn spir_func nofpclass(nan inf) float @sqrt(float noundef nofpclass(nan inf) {{.*}})
      // ROUNDED-DIV-ROUNDED-SQRT-FAST: call reassoc nnan ninf nsz arcp afn float @llvm.fpbuiltin.sqrt.f32(float {{.*}}) #[[ATTR_SQRT:[0-9]+]]
      // LOW-PREC-DIV: call float @llvm.fpbuiltin.sqrt.f32(float {{.*}}) #[[ATTR_SQRT_LOW:[0-9]+]]
      // LOW-PREC-SQRT: call float @llvm.fpbuiltin.sqrt.f32(float {{.*}}) #[[ATTR_SQRT_LOW:[0-9]+]]
      (void)sqrt(Value1);
    });
  });

  deviceQueue.submit([&](handler& cgh) {
    cgh.parallel_for<class KernelFdiv>(numOfItems,
    [=](id<1> wiID) {
      // PREC-SQRT: fdiv float {{.*}}, {{.*}}
      // ROUNDED-SQRT: fdiv float {{.*}}, {{.*}}
      // ROUNDED-SQRT-FAST: call reassoc nnan ninf nsz arcp afn float @llvm.fpbuiltin.fdiv.f32(float {{.*}}) #[[ATTR_DIV:[0-9]+]]
      // PREC-DIV: fdiv float {{.*}}, {{.*}}
      // ROUNDED-DIV: call float @llvm.fpbuiltin.fdiv.f32(float {{.*}}, float {{.*}}) #[[ATTR_DIV:[0-9]+]]
      // ROUNDED-DIV-FAST: call reassoc nnan ninf nsz arcp afn float @llvm.fpbuiltin.fdiv.f32(float {{.*}}, float {{.*}}) #[[ATTR_DIV:[0-9]+]]
      // PREC-FAST: fdiv reassoc nnan ninf nsz arcp afn float {{.*}}, {{.*}}
      // ROUNDED-DIV-ROUNDED-SQRT: call float @llvm.fpbuiltin.fdiv.f32(float {{.*}}, float {{.*}}) #[[ATTR_DIV:[0-9]+]]
      // PREC-SQRT-FAST: call reassoc nnan ninf nsz arcp afn float @llvm.fpbuiltin.fdiv.f32(float {{.*}}, float {{.*}}) #[[ATTR_DIV:[0-9]+]]
      // ROUNDED-SQRT-PREC-DIV: fdiv reassoc nnan ninf nsz arcp afn float {{.*}}, {{.*}}
      // ROUNDED-DIV-PREC-SQRT: call reassoc nnan ninf nsz arcp afn float @llvm.fpbuiltin.fdiv.f32(float {{.*}}, float {{.*}}) #[[ATTR_DIV:[0-9]+]]
      // ROUNDED-DIV-ROUNDED-SQRT-FAST: call reassoc nnan ninf nsz arcp afn float @llvm.fpbuiltin.fdiv.f32(float {{.*}}, float {{.*}}) #[[ATTR_DIV:[0-9]+]]
      // LOW-PREC-DIV: call float @llvm.fpbuiltin.fdiv.f32(float {{.*}}, float {{.*}}) #[[ATTR_FDIV_LOW:[0-9]+]]
      // LOW-PREC-SQRT: fdiv float {{.*}}, {{.*}}
      a[0] = Value1 / Value2;
    });
  });

return 0;
}

// ROUNDED-SQRT: attributes #[[ATTR_SQRT]] = {{.*}}"fpbuiltin-max-error"="3.0"
// ROUNDED-SQRT-FAST: attributes #[[ATTR_SQRT]] = {{.*}}"fpbuiltin-max-error"="3.0"
// ROUNDED-SQRT-FAST: attributes #[[ATTR_DIV]] = {{.*}}"fpbuiltin-max-error"="2.5"
// ROUNDED-DIV: attributes #[[ATTR_DIV]] = {{.*}}"fpbuiltin-max-error"="2.5"
// ROUNDED-DIV-FAST: attributes #[[ATTR_DIV]] = {{.*}}"fpbuiltin-max-error"="2.5"
// ROUNDED-DIV-ROUNDED-SQRT-FAST: attributes #[[ATTR_SQRT]] = {{.*}}"fpbuiltin-max-error"="3.0"
// ROUNDED-DIV-ROUNDED-SQRT-FAST: attributes #[[ATTR_DIV]] = {{.*}}"fpbuiltin-max-error"="2.5"
// LOW-PREC-DIV: attributes #[[ATTR_SQRT_LOW]] = {{.*}}"fpbuiltin-max-error"="1.0"
// LOW-PREC-DIV: attributes #[[ATTR_FDIV_LOW]] = {{.*}}"fpbuiltin-max-error"="2.5"
// LOW-PREC-SQRT: attributes #[[ATTR_SQRT_LOW]] = {{.*}}"fpbuiltin-max-error"="3.0"
