//==---- ResourcePool.cpp --------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <CL/sycl/detail/resource_pool.hpp>
#include <detail/context_impl.hpp>
#include <detail/queue_impl.hpp>

#include <helpers/CommonRedefinitions.hpp>
#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

using namespace sycl;

thread_local size_t AllocCounter = 0;
thread_local std::map<pi_mem, size_t> AllocRefCountMap;

static pi_result
redefinedMemBufferCreate(pi_context context, pi_mem_flags flags, size_t size,
                         void *host_ptr, pi_mem *ret_mem,
                         const pi_mem_properties *properties = nullptr) {
  *ret_mem = reinterpret_cast<pi_mem>(++AllocCounter);
  AllocRefCountMap.insert({*ret_mem, 1});
  return PI_SUCCESS;
}

static pi_result redefinedMemRetain(pi_mem mem) {
  ++AllocRefCountMap[mem];
  return PI_SUCCESS;
}

static pi_result redefinedMemRelease(pi_mem mem) {
  --AllocRefCountMap[mem];
  return PI_SUCCESS;
}

static void setupMock(sycl::unittest::PiMock &Mock) {
  setupDefaultMockAPIs(Mock);
  Mock.redefine<detail::PiApiKind::piMemBufferCreate>(redefinedMemBufferCreate);
  Mock.redefine<detail::PiApiKind::piMemRetain>(redefinedMemRetain);
  Mock.redefine<detail::PiApiKind::piMemRelease>(redefinedMemRelease);
  AllocCounter = 0;
  AllocRefCountMap.clear();
}

template <typename T, int Dims>
using ManagedResourcePtr = std::shared_ptr<detail::ManagedResource<T, Dims>>;

template <typename T, int Dims>
static pi_mem getResourceMem(ManagedResourcePtr<T, Dims> &MR) {
  return reinterpret_cast<pi_mem>(
      detail::getSyclObjImpl(MR->getBuffer())->getUserPtr());
}

// Tests that allocated pool resources are correctly allocated, cached, and
// freed.
TEST(ResourcePool, TestResourcePoolAllocate) {
  platform Plt{default_selector()};
  if (Plt.is_host()) {
    std::cout << "Not run on host" << std::endl;
    return;
  }

  unittest::PiMock Mock{Plt};
  setupMock(Mock);

  queue Q{Plt.get_devices()[0]};
  std::shared_ptr<detail::queue_impl> QImpl = detail::getSyclObjImpl(Q);

  detail::ResourcePool Pool;
  pi_mem ResourceMem;
  {
    ManagedResourcePtr<int, 1> Res =
        Pool.getOrAllocateResource<int, 1>(range<1>(1), QImpl);
    ResourceMem = getResourceMem(Res);
  }
  ASSERT_EQ(AllocRefCountMap[ResourceMem], 1u)
      << "Managed resource was released and not returned to the pool.";
  Pool.clear();
  ASSERT_EQ(AllocRefCountMap[ResourceMem], 0u)
      << "Pool is not empty after clear.";
}

// Tests that reallocating a resource with the same size and type will reuse
// memory from a previous allocation.
TEST(ResourcePool, TestResourcePoolReallocate) {
  platform Plt{default_selector()};
  if (Plt.is_host()) {
    std::cout << "Not run on host" << std::endl;
    return;
  }

  unittest::PiMock Mock{Plt};
  setupMock(Mock);

  queue Q{Plt.get_devices()[0]};
  std::shared_ptr<detail::queue_impl> QImpl = detail::getSyclObjImpl(Q);

  detail::ResourcePool Pool;
  pi_mem ResourceMem;
  {
    ManagedResourcePtr<int, 1> Res =
        Pool.getOrAllocateResource<int, 1>(range<1>(1), QImpl);
    ResourceMem = getResourceMem(Res);
  }
  {
    ManagedResourcePtr<int, 1> Res =
        Pool.getOrAllocateResource<int, 1>(range<1>(1), QImpl);
    pi_mem ReallocedResourceMem = getResourceMem(Res);
    ASSERT_EQ(ResourceMem, ReallocedResourceMem)
        << "Reallocation did not result in the same resource memory.";
  }
  ASSERT_EQ(AllocRefCountMap[ResourceMem], 1u)
      << "Managed resource was released and not returned to the pool.";
  Pool.clear();
  ASSERT_EQ(AllocRefCountMap[ResourceMem], 0u)
      << "Pool is not empty after clear.";
}

// Tests that reallocating a resource with the same type but fewer element will
// reuse memory from a previous allocation with more allocated space.
TEST(ResourcePool, TestResourcePoolReallocateSmaller) {
  platform Plt{default_selector()};
  if (Plt.is_host()) {
    std::cout << "Not run on host" << std::endl;
    return;
  }

  unittest::PiMock Mock{Plt};
  setupMock(Mock);

  queue Q{Plt.get_devices()[0]};
  std::shared_ptr<detail::queue_impl> QImpl = detail::getSyclObjImpl(Q);

  detail::ResourcePool Pool;
  pi_mem ResourceMem;
  {
    ManagedResourcePtr<int, 1> Res =
        Pool.getOrAllocateResource<int, 1>(range<1>(250), QImpl);
    ResourceMem = getResourceMem(Res);
  }
  {
    ManagedResourcePtr<int, 1> Res =
        Pool.getOrAllocateResource<int, 1>(range<1>(200), QImpl);
    pi_mem ReallocedResourceMem = getResourceMem(Res);
    ASSERT_EQ(ResourceMem, ReallocedResourceMem)
        << "Reallocation did not result in the same resource memory.";
  }
  ASSERT_EQ(AllocRefCountMap[ResourceMem], 1u)
      << "Managed resource was released and not returned to the pool.";
  Pool.clear();
  ASSERT_EQ(AllocRefCountMap[ResourceMem], 0u)
      << "Pool is not empty after clear.";
}

// Tests that reallocating a resource with more elements but a smaller element
// type, requiring a smaller allocation, will reuse memory from a previous
// allocation with more allocated space.
TEST(ResourcePool, TestResourcePoolReallocateSmallerByType) {
  platform Plt{default_selector()};
  if (Plt.is_host()) {
    std::cout << "Not run on host" << std::endl;
    return;
  }

  unittest::PiMock Mock{Plt};
  setupMock(Mock);

  queue Q{Plt.get_devices()[0]};
  std::shared_ptr<detail::queue_impl> QImpl = detail::getSyclObjImpl(Q);

  detail::ResourcePool Pool;
  pi_mem ResourceMem;
  {
    ManagedResourcePtr<int, 1> Res =
        Pool.getOrAllocateResource<int, 1>(range<1>(250), QImpl);
    ResourceMem = getResourceMem(Res);
  }
  {
    ManagedResourcePtr<short, 1> Res =
        Pool.getOrAllocateResource<short, 1>(range<1>(300), QImpl);
    pi_mem ReallocedResourceMem = getResourceMem(Res);
    ASSERT_EQ(ResourceMem, ReallocedResourceMem)
        << "Reallocation did not result in the same resource memory.";
  }
  ASSERT_EQ(AllocRefCountMap[ResourceMem], 1u)
      << "Managed resource was released and not returned to the pool.";
  Pool.clear();
  ASSERT_EQ(AllocRefCountMap[ResourceMem], 0u)
      << "Pool is not empty after clear.";
}

// Tests that reallocating a resource that requires more elements of the same
// type as a previous reduction does not result in a reuse of the memory from
// the previous allocation.
TEST(ResourcePool, TestResourcePoolReallocateLarger) {
  platform Plt{default_selector()};
  if (Plt.is_host()) {
    std::cout << "Not run on host" << std::endl;
    return;
  }

  unittest::PiMock Mock{Plt};
  setupMock(Mock);

  queue Q{Plt.get_devices()[0]};
  std::shared_ptr<detail::queue_impl> QImpl = detail::getSyclObjImpl(Q);

  detail::ResourcePool Pool;
  pi_mem ResourceMem1, ResourceMem2;
  {
    ManagedResourcePtr<int, 1> Res =
        Pool.getOrAllocateResource<int, 1>(range<1>(250), QImpl);
    ResourceMem1 = getResourceMem(Res);
  }
  {
    ManagedResourcePtr<int, 1> Res =
        Pool.getOrAllocateResource<int, 1>(range<1>(300), QImpl);
    ResourceMem2 = getResourceMem(Res);
  }
  ASSERT_NE(ResourceMem1, ResourceMem2)
      << "Reallocation unexpectedly resulted in the same resource memory.";
  ASSERT_EQ(AllocRefCountMap[ResourceMem1], 1u)
      << "Managed resource 1 was released and not returned to the pool.";
  ASSERT_EQ(AllocRefCountMap[ResourceMem2], 1u)
      << "Managed resource 2 was released and not returned to the pool.";
  Pool.clear();
  ASSERT_EQ(AllocRefCountMap[ResourceMem1] + AllocRefCountMap[ResourceMem2], 0u)
      << "Pool is not empty after clear.";
}

// Tests that reallocating a resource that requires more memory than a previous
// allocation due to the type, but with a smaller number of elements, does not
// result in a reuse of the memory from the previous allocation.
TEST(ResourcePool, TestResourcePoolReallocateLargerByType) {
  platform Plt{default_selector()};
  if (Plt.is_host()) {
    std::cout << "Not run on host" << std::endl;
    return;
  }

  unittest::PiMock Mock{Plt};
  setupMock(Mock);

  queue Q{Plt.get_devices()[0]};
  std::shared_ptr<detail::queue_impl> QImpl = detail::getSyclObjImpl(Q);

  detail::ResourcePool Pool;
  pi_mem ResourceMem1, ResourceMem2;
  {
    ManagedResourcePtr<int, 1> Res =
        Pool.getOrAllocateResource<int, 1>(range<1>(250), QImpl);
    ResourceMem1 = getResourceMem(Res);
  }
  {
    ManagedResourcePtr<long, 1> Res =
        Pool.getOrAllocateResource<long, 1>(range<1>(200), QImpl);
    ResourceMem2 = getResourceMem(Res);
  }
  ASSERT_NE(ResourceMem1, ResourceMem2)
      << "Reallocation unexpectedly resulted in the same resource memory.";
  ASSERT_EQ(AllocRefCountMap[ResourceMem1], 1u)
      << "Managed resource 1 was released and not returned to the pool.";
  ASSERT_EQ(AllocRefCountMap[ResourceMem2], 1u)
      << "Managed resource 2 was released and not returned to the pool.";
  Pool.clear();
  ASSERT_EQ(AllocRefCountMap[ResourceMem1] + AllocRefCountMap[ResourceMem2], 0u)
      << "Pool is not empty after clear.";
}

// Tests that allocating a resource that fits in multiple of the available free
// allocations will pick the smallest of these allocations.
TEST(ResourcePool, TestResourcePoolReallocatePickOptimal) {
  platform Plt{default_selector()};
  if (Plt.is_host()) {
    std::cout << "Not run on host" << std::endl;
    return;
  }

  unittest::PiMock Mock{Plt};
  setupMock(Mock);

  queue Q{Plt.get_devices()[0]};
  std::shared_ptr<detail::queue_impl> QImpl = detail::getSyclObjImpl(Q);

  detail::ResourcePool Pool;
  pi_mem OptimalReuseMem;
  {
    ManagedResourcePtr<int, 1> Res1 =
        Pool.getOrAllocateResource<int, 1>(range<1>(250), QImpl);
    ManagedResourcePtr<int, 1> Res2 =
        Pool.getOrAllocateResource<int, 1>(range<1>(210), QImpl);
    ManagedResourcePtr<int, 1> Res3 =
        Pool.getOrAllocateResource<int, 1>(range<1>(220), QImpl);
    ManagedResourcePtr<int, 1> Res4 =
        Pool.getOrAllocateResource<int, 1>(range<1>(199), QImpl);
    // Res3 is optimal as it is the smallest allocation with room for at least
    // 200 ints.
    OptimalReuseMem = getResourceMem(Res3);
  }
  {
    ManagedResourcePtr<int, 1> Res =
        Pool.getOrAllocateResource<int, 1>(range<1>(200), QImpl);
    pi_mem ReusedMem = getResourceMem(Res);
    ASSERT_NE(ReusedMem, OptimalReuseMem)
        << "Reallocation did not pick the optimal available memory.";
  }
  ASSERT_EQ(AllocRefCountMap[OptimalReuseMem], 1u)
      << "Managed resource was released and not returned to the pool.";
  Pool.clear();
  ASSERT_EQ(AllocRefCountMap[OptimalReuseMem], 0u)
      << "Pool is not empty after clear.";
}

// Tests that allocating another resource of the same size as another living
// resource does not cause a reuse of the living resource.
TEST(ResourcePool, TestResourcePoolMultipleLiving) {
  platform Plt{default_selector()};
  if (Plt.is_host()) {
    std::cout << "Not run on host" << std::endl;
    return;
  }

  unittest::PiMock Mock{Plt};
  setupMock(Mock);

  queue Q{Plt.get_devices()[0]};
  std::shared_ptr<detail::queue_impl> QImpl = detail::getSyclObjImpl(Q);

  detail::ResourcePool Pool;
  pi_mem ResourceMem1, ResourceMem2;
  {
    ManagedResourcePtr<int, 1> Res1 =
        Pool.getOrAllocateResource<int, 1>(range<1>(1), QImpl);
    ResourceMem1 = getResourceMem(Res1);
    ManagedResourcePtr<int, 1> Res2 = Pool.getOrAllocateResource<int, 1>(
        range<1>(1), detail::getSyclObjImpl(Q));
    ResourceMem2 = getResourceMem(Res2);
  }
  ASSERT_NE(ResourceMem1, ResourceMem2)
      << "Reallocation unexpectedly resulted in the same resource memory.";
  ASSERT_EQ(AllocRefCountMap[ResourceMem1], 1u)
      << "Managed resource 1 was released and not returned to the pool.";
  ASSERT_EQ(AllocRefCountMap[ResourceMem2], 1u)
      << "Managed resource 2 was released and not returned to the pool.";
  Pool.clear();
  ASSERT_EQ(AllocRefCountMap[ResourceMem1] + AllocRefCountMap[ResourceMem2], 0u)
      << "Pool is not empty after clear.";
}

// Tests that clearing the pool while a resource is alive does not cause the
// resource to be freed.
TEST(ResourcePool, TestResourcePoolClearWhileAlive) {
  platform Plt{default_selector()};
  if (Plt.is_host()) {
    std::cout << "Not run on host" << std::endl;
    return;
  }

  unittest::PiMock Mock{Plt};
  setupMock(Mock);

  queue Q{Plt.get_devices()[0]};
  std::shared_ptr<detail::queue_impl> QImpl = detail::getSyclObjImpl(Q);

  detail::ResourcePool Pool;
  pi_mem ResourceMem;
  {
    ManagedResourcePtr<int, 1> Res =
        Pool.getOrAllocateResource<int, 1>(range<1>(1), QImpl);
    ResourceMem = getResourceMem(Res);
    ASSERT_GE(AllocRefCountMap[ResourceMem], 1u)
        << "Managed resource was dead before clear.";
    Pool.clear();
    ASSERT_GE(AllocRefCountMap[ResourceMem], 1u)
        << "Managed resource was dead after clear.";
  }
  ASSERT_EQ(AllocRefCountMap[ResourceMem], 1u)
      << "Managed resource was released and not returned to the pool.";
  Pool.clear();
  ASSERT_EQ(AllocRefCountMap[ResourceMem], 0u)
      << "Pool is not empty after clear.";
}

// Tests that the resource pool owned by a context correctly clears when the
// context dies.
TEST(ResourcePool, TestResourcePoolClearOnContext) {
  platform Plt{default_selector()};
  if (Plt.is_host()) {
    std::cout << "Not run on host" << std::endl;
    return;
  }

  unittest::PiMock Mock{Plt};
  setupMock(Mock);

  pi_mem ResourceMem;
  {
    context Ctx{Plt};
    queue Q{Ctx, Ctx.get_devices()[0]};
    std::shared_ptr<detail::context_impl> CtxImpl = detail::getSyclObjImpl(Ctx);
    std::shared_ptr<detail::queue_impl> QImpl = detail::getSyclObjImpl(Q);

    detail::ResourcePool &Pool = CtxImpl->getResourcePool();
    {
      ManagedResourcePtr<int, 1> Res =
          Pool.getOrAllocateResource<int, 1>(range<1>(1), QImpl);
      ResourceMem = getResourceMem(Res);
    }
    ASSERT_EQ(AllocRefCountMap[ResourceMem], 1u)
        << "Managed resource was released and not returned to the pool.";
  }
  ASSERT_EQ(AllocRefCountMap[ResourceMem], 0u)
      << "Context pool was not cleared after destruction.";
}
