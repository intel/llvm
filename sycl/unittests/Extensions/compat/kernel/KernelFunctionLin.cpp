/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Ltd.
 *
 *  Part of the LLVM Project, under the Apache License v2.0 with LLVM
 *  Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  SYCL compatibility API
 *
 *  KernelFunctionLin.cpp
 *
 *  Description:
 *    kernel_function header API tests
 **************************************************************************/

// The original source was under the license below:
// ====------ KernelFunctionLin.cpp---------- -*- C++ -* ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <dlfcn.h>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <sycl/ext/oneapi/experimental/compat.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi::experimental;

template <class T> void testTemplateKernel(T *data) {}

void testKernel(void *data) {}

template <class T> int getTemplateFuncAttrs() {
  // test_feature:kernel_function_info
  compat::kernel_function_info attrs;
  // test_feature:get_kernel_function_info
  compat::get_kernel_function_info(&attrs, (const void *)testTemplateKernel<T>);

  int threadPerBlock = attrs.max_work_group_size;

  return threadPerBlock;
}

int getFuncAttrs() {
  // test_feature:kernel_function_info
  compat::kernel_function_info attrs;
  // test_feature:get_kernel_function_info
  compat::get_kernel_function_info(&attrs, (const void *)testKernel);

  int threadPerBlock = attrs.max_work_group_size;

  return threadPerBlock;
}

TEST(kernel_functor, kernel_functor_ptr) {
  compat::device_ext &dev_ct1 = compat::get_current_device();
  sycl::queue *q_ct1 = dev_ct1.default_queue();

  int Size = dev_ct1.get_info<sycl::info::device::max_work_group_size>();
  EXPECT_EQ(getTemplateFuncAttrs<int>(), Size);
  EXPECT_EQ(getFuncAttrs(), Size);

  void *M;
  // test_feature:kernel_functor
  compat::kernel_functor F;

  M = dlopen("./libExtensionscompat_kernel_function_test.so", RTLD_LAZY);
  if (M == NULL) {
    std::cout << "Could not load the library" << '\n';
    FAIL();
  }

  std::string FunctionName = "foo_wrapper";
  F = (compat::kernel_functor)dlsym(M, FunctionName.c_str());
  if (F == NULL) {
    std::cout << "Could not load function pointer" << '\n';
    dlclose(M);
    FAIL();
  }

  int sharedSize = 10;
  void **param = nullptr, **extra = nullptr;

  int *dev = sycl::malloc_shared<int>(16, *q_ct1);
  for (int i = 0; i < 16; i++) {
    dev[i] = 0;
  }
  param = (void **)(&dev);
  F(*q_ct1,
    sycl::nd_range<3>(sycl::range<3>(1, 1, 2) * sycl::range<3>(1, 1, 8),
                      sycl::range<3>(1, 1, 8)),
    sharedSize, param, extra);
  q_ct1->wait_and_throw();

  for (int i = 0; i < 16; i++) {
    EXPECT_EQ(dev[i], i);
  }

  sycl::free(dev, *q_ct1);
  dlclose(M);
}
