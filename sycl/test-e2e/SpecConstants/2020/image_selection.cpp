// REQUIRES: opencl, level-zero, gpu, ocloc

// Check the case when -fsycl-add-default-spec-consts-image option is used which
// results in generation of two types of images: where specialization constants
// are replaced with defaults and original images.

// clang-format off
// RUN: %clangxx -fsycl-add-default-spec-consts-image -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts %s -o %t1.out
// RUN: env SYCL_PI_TRACE=-1 %{run} %t1.out | FileCheck --match-full-lines --check-prefix=CHECK-ENABLED %s
// clang-format on

// Check the behaviour when -fsycl-add-default-spec-consts-image option is not
// used.

// RUN: %clangxx  -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts %s -o %t2.out
// RUN: env SYCL_PI_TRACE=-1 %{run} %t2.out | FileCheck --match-full-lines --check-prefix=CHECK-DEFAULT %s

// Check the behaviour when -fsycl-add-default-spec-consts-image option is used
// and we have spirv image in addition to AOT.

// clang-format off
// RUN: %clangxx  -fsycl -fsycl-targets=spir64,spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts %s -o %t3.out
// RUN: env SYCL_PI_TRACE=-1 %{run} %t3.out | FileCheck --match-full-lines --check-prefix=CHECK-MIX %s
// clang-format on

// Check the behaviour when -fsycl-add-default-spec-consts-image option is used
// and default value is explicitly set with the same value - we are supposed to
// choose images with inlined values in this case.

// clang-format off
// RUN: %clangxx  -fsycl-add-default-spec-consts-image -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts %s -o %t3.out
// RUN: env SYCL_PI_TRACE=-1 %{run} %t3.out | FileCheck --match-full-lines --check-prefix=CHECK-DEFAULT-EXPLICIT-SET %s
// clang-format on

// Check the behaviour when -fsycl-add-default-spec-consts-image option is used
// and value of specialization constant is changed to new value and then back to
// the default value - we are supposed to choose images with inlined values in
// this case.

// clang-format off
// RUN: %clangxx  -fsycl-add-default-spec-consts-image -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts %s -o %t3.out
// RUN: env SYCL_PI_TRACE=-1 %{run} %t3.out | FileCheck --match-full-lines --check-prefix=CHECK-DEFAULT-BACK-TO-DEFAULT %s
// clang-format on

#include <sycl/sycl.hpp>

constexpr sycl::specialization_id<int> int_id(3);

class Kernel1;
class Kernel2;
class Kernel3;
class Kernel4;

int main() {
  sycl::queue Q;
  int *Res = sycl::malloc_host<int>(1, Q);

  // Test the case when we set or don't set specialization constant for
  // invocations of the same kernel depending on iteration number in the loop.
  // It allows to check that we select the right device image for kernel
  // submission depending on whether spec const value was set or not. a. In the
  // case when we select image where specialization constants are replaced with
  // default value - specialization constant buffer is not created and we set
  // nullptr in piextKernelSetArgMemObj (4th parameter) b. In the case when we
  // select regular image - specialization constant buffer is created and we set
  // a real pointer in piextKernelSetArgMemObj.

  // CHECK-DEFAULT: Submission 0
  // CHECK-DEFAULT: ---> piextKernelSetArgMemObj(
  // CHECK-DEFAULT-NEXT:	<unknown> : {{.*}}
  // CHECK-DEFAULT-NEXT:	<unknown> : {{.*}}
  // CHECK-DEFAULT-NEXT:	<unknown> : {{.*}}
  // CHECK-DEFAULT-NEXT:	<unknown> : 0x{{[0-9,a-f]+}}
  // CHECK-DEFAULT-NEXT: ) ---> 	pi_result : PI_SUCCESS
  // CHECK-DEFAULT: Default value of specialization constant was used.

  // CHECK-DEFAULT: Submission 1
  // CHECK-DEFAULT: ---> piextKernelSetArgMemObj(
  // CHECK-DEFAULT-NEXT:	<unknown> : {{.*}}
  // CHECK-DEFAULT-NEXT:	<unknown> : {{.*}}
  // CHECK-DEFAULT-NEXT:	<unknown> : {{.*}}
  // CHECK-DEFAULT-NEXT:	<unknown> : 0x{{[0-9,a-f]+}}
  // CHECK-DEFAULT-NEXT: ) ---> 	pi_result : PI_SUCCESS
  // CHECK-DEFAULT: New specialization constant value was set.

  // CHECK-DEFAULT: Submission 2
  // CHECK-DEFAULT: ---> piextKernelSetArgMemObj(
  // CHECK-DEFAULT-NEXT:	<unknown> : {{.*}}
  // CHECK-DEFAULT-NEXT:	<unknown> : {{.*}}
  // CHECK-DEFAULT-NEXT:	<unknown> : {{.*}}
  // CHECK-DEFAULT-NEXT:	<unknown> : 0x{{[0-9,a-f]+}}
  // CHECK-DEFAULT-NEXT: ) ---> 	pi_result : PI_SUCCESS
  // CHECK-DEFAULT: Default value of specialization constant was used.

  // CHECK-DEFAULT: Submission 3
  // CHECK-DEFAULT: ---> piextKernelSetArgMemObj(
  // CHECK-DEFAULT-NEXT:	<unknown> : {{.*}}
  // CHECK-DEFAULT-NEXT:	<unknown> : {{.*}}
  // CHECK-DEFAULT-NEXT:	<unknown> : {{.*}}
  // CHECK-DEFAULT-NEXT:	<unknown> : 0x{{[0-9,a-f]+}}
  // CHECK-DEFAULT-NEXT: ) ---> 	pi_result : PI_SUCCESS
  // CHECK-DEFAULT: New specialization constant value was set.

  // CHECK-ENABLED: Submission 0
  // CHECK-ENABLED: ---> piextKernelSetArgMemObj(
  // CHECK-ENABLED-NEXT:	<unknown> : {{.*}}
  // CHECK-ENABLED-NEXT:	<unknown> : {{.*}}
  // CHECK-ENABLED-NEXT:	<unknown> : {{.*}}
  // CHECK-ENABLED-NEXT:	<unknown> : 0
  // CHECK-ENABLED-NEXT: ) ---> 	pi_result : PI_SUCCESS
  // CHECK-ENABLED: Default value of specialization constant was used.

  // CHECK-ENABLED: Submission 1
  // CHECK-ENABLED: ---> piextKernelSetArgMemObj(
  // CHECK-ENABLED-NEXT:	<unknown> : {{.*}}
  // CHECK-ENABLED-NEXT:	<unknown> : {{.*}}
  // CHECK-ENABLED-NEXT:	<unknown> : {{.*}}
  // CHECK-ENABLED-NEXT:	<unknown> : 0x{{[0-9,a-f]+}}
  // CHECK-ENABLED-NEXT: ) ---> 	pi_result : PI_SUCCESS
  // CHECK-ENABLED: New specialization constant value was set.

  // CHECK-ENABLED: Submission 2
  // CHECK-ENABLED: ---> piextKernelSetArgMemObj(
  // CHECK-ENABLED-NEXT:	<unknown> : {{.*}}
  // CHECK-ENABLED-NEXT:	<unknown> : {{.*}}
  // CHECK-ENABLED-NEXT:	<unknown> : {{.*}}
  // CHECK-ENABLED-NEXT:	<unknown> : 0
  // CHECK-ENABLED-NEXT: ) ---> 	pi_result : PI_SUCCESS
  // CHECK-ENABLED: Default value of specialization constant was used.

  // CHECK-ENABLED: Submission 3
  // CHECK-ENABLED: ---> piextKernelSetArgMemObj(
  // CHECK-ENABLED-NEXT:	<unknown> : {{.*}}
  // CHECK-ENABLED-NEXT:	<unknown> : {{.*}}
  // CHECK-ENABLED-NEXT:	<unknown> : {{.*}}
  // CHECK-ENABLED-NEXT:	<unknown> : 0x{{[0-9,a-f]+}}
  // CHECK-ENABLED-NEXT: ) ---> 	pi_result : PI_SUCCESS
  // CHECK-ENABLED: New specialization constant value was set.

  // CHECK-MIX: Submission 0
  // CHECK-MIX: Default value of specialization constant was used.

  // CHECK-MIX: Submission 1
  // CHECK-MIX: New specialization constant value was set.

  // CHECK-MIX: Submission 2
  // CHECK-MIX: Default value of specialization constant was used.

  // CHECK-MIX: Submission 3
  // CHECK-MIX: New specialization constant value was set.
  for (int i = 0; i < 4; i++) {
    std::cout << std::endl << "Submission " << i << std::endl;
    Q.submit([&](sycl::handler &cgh) {
       // Set spec const only when odd number.
       if (i % 2)
         cgh.set_specialization_constant<int_id>(5);

       cgh.single_task<Kernel1>([=](sycl::kernel_handler h) {
         auto SpecConst = h.get_specialization_constant<int_id>();
         *Res = SpecConst == 3 ? 0 : 1;
       });
     }).wait();

    if (*Res)
      std::cout << "New specialization constant value was set." << std::endl;
    else
      std::cout << "Default value of specialization constant was used."
                << std::endl;
  }

  // Test that the same works when kernel_bundle is used.
  // In this we don't set specialization constant value for bundle, so default
  // value is used and SYCL RT selects image where values are replaced with
  // default, that's why nullptr is set as 4th parameter of
  // piextKernelSetArgMemObj.
  // CHECK-DEFAULT: Kernel bundle
  // CHECK-DEFAULT: ---> piextKernelSetArgMemObj(
  // CHECK-DEFAULT-NEXT:	<unknown> : {{.*}}
  // CHECK-DEFAULT-NEXT:	<unknown> : {{.*}}
  // CHECK-DEFAULT-NEXT:	<unknown> : {{.*}}
  // CHECK-DEFAULT-NEXT:	<unknown> : 0x{{[0-9,a-f]+}}
  // CHECK-DEFAULT-NEXT: ) ---> 	pi_result : PI_SUCCESS
  // CHECK-DEFAULT: Default value of specialization constant was used.

  // CHECK-ENABLED: Kernel bundle
  // CHECK-ENABLED: ---> piextKernelSetArgMemObj(
  // CHECK-ENABLED-NEXT:	<unknown> : {{.*}}
  // CHECK-ENABLED-NEXT:	<unknown> : {{.*}}
  // CHECK-ENABLED-NEXT:	<unknown> : {{.*}}
  // CHECK-ENABLED-NEXT:	<unknown> : 0
  // CHECK-ENABLED-NEXT: ) ---> 	pi_result : PI_SUCCESS
  // CHECK-ENABLED: Default value of specialization constant was used.

  // CHECK-MIX: Kernel bundle
  // CHECK-MIX: Default value of specialization constant was used.
  std::cout << "Kernel bundle" << std::endl;
  sycl::kernel_id kernelId = sycl::get_kernel_id<Kernel2>();
  auto Bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      Q.get_context(), {kernelId});
  Q.submit([&](sycl::handler &cgh) {
     cgh.use_kernel_bundle(Bundle);
     cgh.single_task<Kernel2>([=](sycl::kernel_handler h) {
       auto SpecConst = h.get_specialization_constant<int_id>();
       *Res = SpecConst == 3 ? 0 : 1;
     });
   }).wait();

  if (*Res)
    std::cout << "New specialization constant value was set." << std::endl;
  else
    std::cout << "Default value of specialization constant was used."
              << std::endl;

  // Test that if user calls set_specialization_constant with the value equal to
  // default then we choose image with inlined default values of specialization
  // constants. We are verifying that by checking the 4th parameter is set to
  // zero.
  // CHECK-DEFAULT-EXPLICIT-SET: Default value was explicitly set
  // CHECK-DEFAULT-EXPLICIT-SET: ---> piextKernelSetArgMemObj(
  // CHECK-DEFAULT-EXPLICIT-SET-NEXT:	<unknown> : {{.*}}
  // CHECK-DEFAULT-EXPLICIT-SET-NEXT:	<unknown> : {{.*}}
  // CHECK-DEFAULT-EXPLICIT-SET-NEXT:	<unknown> : {{.*}}
  // CHECK-DEFAULT-EXPLICIT-SET-NEXT:	<unknown> : 0
  // CHECK-DEFAULT-EXPLICIT-SET-NEXT: ) ---> 	pi_result : PI_SUCCESS
  // CHECK-DEFAULT-EXPLICIT-SET: Default value of specialization constant was used.
  std::cout << "Default value was explicitly set" << std::endl;
  Q.submit([&](sycl::handler &cgh) {
     cgh.set_specialization_constant<int_id>(3);

     cgh.single_task<Kernel3>([=](sycl::kernel_handler h) {
       auto SpecConst = h.get_specialization_constant<int_id>();
       *Res = SpecConst == 3 ? 0 : 1;
     });
   }).wait();

  if (*Res)
    std::cout << "New specialization constant value was set." << std::endl;
  else
    std::cout << "Default value of specialization constant was used."
              << std::endl;

  // Test that if user sets new value of specialization constant and then
  // changes it back to default value then we choose image with inlined default
  // values of specialization constants. We are verifying that by checking the
  // 4th parameter is set to zero.
  // CHECK-DEFAULT-BACK-TO-DEFAULT: Changed to new value and then default value was explicitly set
  // CHECK-DEFAULT-BACK-TO-DEFAULT: ---> piextKernelSetArgMemObj(
  // CHECK-DEFAULT-BACK-TO-DEFAULT-NEXT:	<unknown> : {{.*}}
  // CHECK-DEFAULT-BACK-TO-DEFAULT-NEXT:	<unknown> : {{.*}}
  // CHECK-DEFAULT-BACK-TO-DEFAULT-NEXT:	<unknown> : {{.*}}
  // CHECK-DEFAULT-BACK-TO-DEFAULT-NEXT:	<unknown> : 0
  // CHECK-DEFAULT-BACK-TO-DEFAULT-NEXT: ) ---> 	pi_result : PI_SUCCESS
  // CHECK-DEFAULT-BACK-TO-DEFAULT: Default value of specialization constant was used.
  std::cout << "Changed to new value and then default value was explicitly set"
            << std::endl;
  Q.submit([&](sycl::handler &cgh) {
     cgh.set_specialization_constant<int_id>(4);
     cgh.set_specialization_constant<int_id>(3);

     cgh.single_task<Kernel4>([=](sycl::kernel_handler h) {
       auto SpecConst = h.get_specialization_constant<int_id>();
       *Res = SpecConst == 3 ? 0 : 1;
     });
   }).wait();

  if (*Res)
    std::cout << "New specialization constant value was set." << std::endl;
  else
    std::cout << "Default value of specialization constant was used."
              << std::endl;

  return 0;
}
