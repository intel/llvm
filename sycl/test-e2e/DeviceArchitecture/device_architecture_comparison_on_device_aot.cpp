// REQUIRES: arch-intel_gpu_pvc, ocloc
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_pvc %s -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#include <sycl/usm.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;

int main() {
  int *result;
  int N = 29;

  {
    queue q(gpu_selector_v);

    result = (int *)malloc_shared(N * sizeof(int), q);

    q.submit([&](handler &cgh) { cgh.fill(result, 0, N); }).wait();

    q.submit([&](handler &cgh) {
      cgh.single_task([=]() {
        // test if_architecture_is with category
        if_architecture_is<arch_category::intel_gpu>([&]() {
          result[0] = 1;
        }).otherwise([&]() {});

        // negative test if_architecture_is with category
        if_architecture_is<arch_category::nvidia_gpu>([&]() {}).otherwise(
            [&]() { result[1] = 1; });

        // test else_if_architecture_is with category - 1
        if_architecture_is<arch_category::nvidia_gpu>([&]() {})
            .else_if_architecture_is<arch_category::intel_gpu>(
                [&]() { result[2] = 1; })
            .otherwise([&]() {});

        // test else_if_architecture_is with category - 2
        if_architecture_is<arch_category::nvidia_gpu>([&]() {})
            .else_if_architecture_is<arch_category::amd_gpu>([&]() {})
            .else_if_architecture_is<arch_category::intel_gpu>(
                [&]() { result[3] = 1; })
            .otherwise([&]() {});

        // test if_architecture_is_lt
        if_architecture_is_lt<architecture::intel_gpu_pvc_vg>([&]() {
          result[4] = 1;
        }).otherwise([&]() {});

        // negative test if_architecture_is_lt - 1
        if_architecture_is_lt<architecture::intel_gpu_pvc>([&]() {}).otherwise(
            [&]() { result[5] = 1; });

        // negative test if_architecture_is_lt - 2
        if_architecture_is_lt<architecture::nvidia_gpu_sm_89>([&]() {
        }).otherwise([&]() { result[6] = 1; });

        // test else_if_architecture_is_lt - 1
        if_architecture_is_lt<architecture::intel_gpu_bdw>([&]() {})
            .else_if_architecture_is_lt<architecture::intel_gpu_pvc_vg>(
                [&]() { result[7] = 1; })
            .otherwise([&]() {});

        // test else_if_architecture_is_lt - 2
        if_architecture_is_lt<architecture::intel_gpu_bdw>([&]() {
        }).else_if_architecture_is_lt<architecture::nvidia_gpu_sm_89>([&]() {})
            .else_if_architecture_is_lt<architecture::intel_gpu_pvc_vg>(
                [&]() { result[8] = 1; })
            .otherwise([&]() {});

        // test if_architecture_is_le
        if_architecture_is_le<architecture::intel_gpu_pvc>([&]() {
          result[9] = 1;
        }).otherwise([&]() {});

        // negative test if_architecture_is_le - 1
        if_architecture_is_le<architecture::intel_gpu_skl>([&]() {}).otherwise(
            [&]() { result[10] = 1; });

        // negative test if_architecture_is_le - 2
        if_architecture_is_le<architecture::nvidia_gpu_sm_89>([&]() {
        }).otherwise([&]() { result[11] = 1; });

        // test else_if_architecture_is_le - 1
        if_architecture_is_le<architecture::intel_gpu_skl>([&]() {})
            .else_if_architecture_is_le<architecture::intel_gpu_pvc>(
                [&]() { result[12] = 1; })
            .otherwise([&]() {});

        // test else_if_architecture_is_le - 2
        if_architecture_is_le<architecture::intel_gpu_skl>([&]() {
        }).else_if_architecture_is_le<architecture::nvidia_gpu_sm_89>([&]() {})
            .else_if_architecture_is_le<architecture::intel_gpu_pvc>(
                [&]() { result[13] = 1; })
            .otherwise([&]() {});

        // test if_architecture_is_gt
        if_architecture_is_gt<architecture::intel_gpu_skl>([&]() {
          result[14] = 1;
        }).otherwise([&]() {});

        // negative test if_architecture_is_gt - 1
        if_architecture_is_gt<architecture::intel_gpu_pvc_vg>([&]() {
        }).otherwise([&]() { result[15] = 1; });

        // negative test if_architecture_is_gt - 2
        if_architecture_is_gt<architecture::nvidia_gpu_sm_89>([&]() {
        }).otherwise([&]() { result[16] = 1; });

        // test else_if_architecture_is_gt - 1
        if_architecture_is_gt<architecture::intel_gpu_pvc_vg>([&]() {})
            .else_if_architecture_is_gt<architecture::intel_gpu_skl>(
                [&]() { result[17] = 1; })
            .otherwise([&]() {});

        // test else_if_architecture_is_gt - 2
        if_architecture_is_gt<architecture::intel_gpu_pvc_vg>([&]() {
        }).else_if_architecture_is_gt<architecture::nvidia_gpu_sm_89>([&]() {})
            .else_if_architecture_is_gt<architecture::intel_gpu_skl>(
                [&]() { result[18] = 1; })
            .otherwise([&]() {});

        // test if_architecture_is_ge
        if_architecture_is_ge<architecture::intel_gpu_pvc>([&]() {
          result[19] = 1;
        }).otherwise([&]() {});

        // negative test if_architecture_is_ge - 1
        if_architecture_is_ge<architecture::intel_gpu_pvc_vg>([&]() {
        }).otherwise([&]() { result[20] = 1; });

        // negative test if_architecture_is_ge - 2
        if_architecture_is_ge<architecture::nvidia_gpu_sm_89>([&]() {
        }).otherwise([&]() { result[21] = 1; });

        // test else_if_architecture_is_ge - 1
        if_architecture_is_ge<architecture::intel_gpu_pvc_vg>([&]() {})
            .else_if_architecture_is_ge<architecture::intel_gpu_pvc>(
                [&]() { result[22] = 1; })
            .otherwise([&]() {});

        // test else_if_architecture_is_ge - 2
        if_architecture_is_ge<architecture::intel_gpu_pvc_vg>([&]() {
        }).else_if_architecture_is_ge<architecture::nvidia_gpu_sm_89>([&]() {})
            .else_if_architecture_is_ge<architecture::intel_gpu_pvc>(
                [&]() { result[23] = 1; })
            .otherwise([&]() {});

        // test if_architecture_is_between
        if_architecture_is_between<architecture::intel_gpu_skl,
                                   architecture::intel_gpu_pvc_vg>([&]() {
          result[24] = 1;
        }).otherwise([&]() {});

        // negative test if_architecture_is_between - 1
        if_architecture_is_between<architecture::intel_gpu_dg1,
                                   architecture::intel_gpu_dg2_g12>([&]() {
        }).otherwise([&]() { result[25] = 1; });

        // negative test if_architecture_is_between - 2
        if_architecture_is_between<architecture::intel_gpu_dg1,
                                   architecture::nvidia_gpu_sm_89>([&]() {
        }).otherwise([&]() { result[26] = 1; });

        // test else_if_architecture_is_between - 1
        if_architecture_is_between<architecture::intel_gpu_dg1,
                                   architecture::intel_gpu_dg2_g12>([&]() {})
            .else_if_architecture_is_between<architecture::intel_gpu_skl,
                                             architecture::intel_gpu_pvc>(
                [&]() { result[27] = 1; })
            .otherwise([&]() {});

        // test else_if_architecture_is_between - 2
        if_architecture_is_between<architecture::intel_gpu_dg1,
                                   architecture::intel_gpu_dg2_g12>([&]() {})
            .else_if_architecture_is_between<architecture::intel_gpu_dg1,
                                             architecture::nvidia_gpu_sm_89>(
                [&]() {})
            .else_if_architecture_is_between<architecture::intel_gpu_skl,
                                             architecture::intel_gpu_pvc>(
                [&]() { result[28] = 1; })
            .otherwise([&]() {});

        // if adding new test here, don't forget to increment result's index and
        // value of N variable
      });
    });
  }

  bool failed = false;
  for (int i = 0; i < N; i++)
    if (result[i] != 1) {
      std::cout << "Verification of the test " << i << " failed." << std::endl;
      failed = true;
    }

  return failed;
}
