#include "detail/kernel_impl.hpp"
#include <cstring>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>

#include <sycl/detail/defines_elementary.hpp>

#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/TestKernel.hpp>
#include <helpers/UrMock.hpp>

#include <detail/context_impl.hpp>

namespace syclex = sycl::ext::oneapi::experimental;
const auto KernelID = sycl::get_kernel_id<TestKernel<>>();

inline ur_result_t redefine_urKernelGetGroupInfo_Success(void *pParams) {
  auto params = reinterpret_cast<ur_kernel_get_group_info_params_t *>(pParams);
  switch (*params->ppropName) {
  case UR_KERNEL_GROUP_INFO_WORK_GROUP_SIZE: {
    if (*params->ppPropValue) {
      auto RealVal = reinterpret_cast<size_t *>(*params->ppPropValue);
      RealVal[0] = 123;
    }
    return UR_RESULT_SUCCESS;
  }
  case UR_KERNEL_GROUP_INFO_GLOBAL_WORK_SIZE: {
    if (*params->ppPropValue) {
      auto RealVal = reinterpret_cast<size_t *>(*params->ppPropValue);
      RealVal[0] = 123;
      RealVal[1] = 213;
      RealVal[2] = 321;
    }
    if (*params->ppPropSizeRet)
      **params->ppPropSizeRet = 3 * sizeof(size_t);
    return UR_RESULT_SUCCESS;
  }
  default: {
    return UR_RESULT_SUCCESS;
  }
  }
}

inline ur_result_t redefine_urKernelGetGroupInfo_Unsupported(void *pParams) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

inline ur_result_t redefine_urKernelGetGroupInfo_Exception(void *pParams) {
  return UR_RESULT_ERROR_INVALID_ARGUMENT;
}

const auto getQueue() {
  sycl::platform Plt = sycl::platform();
  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  sycl::queue Queue{
    Ctx, Dev, sycl::property_list{sycl::property::queue::enable_profiling{}}};
  return Queue;
}

const auto getKernel(const sycl::queue& Q)
{
  auto KernelBundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
    Q.get_context(), std::vector<sycl::kernel_id>{KernelID});
  return KernelBundle.get_kernel(KernelID);
}

TEST(LaunchQueries, GetWorkGroupSizeSuccess)
{
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_replace_callback("urKernelGetGroupInfo",
                                          &redefine_urKernelGetGroupInfo_Success);
  const auto Queue = getQueue();
  const auto Kernel = getKernel(Queue);
  const auto maxWorkGroupSize = Kernel.template ext_oneapi_get_info<
    syclex::info::kernel_queue_specific::max_work_group_size>(Queue);
  const auto result = std::is_same_v<std::remove_cv_t<decltype(maxWorkGroupSize)>, size_t>;
  ASSERT_TRUE(result);
  ASSERT_EQ(maxWorkGroupSize, 123);
}

TEST(LaunchQueries, GetWorkGroupSizeUnsupported)
{
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_replace_callback("urKernelGetGroupInfo",
                                          &redefine_urKernelGetGroupInfo_Unsupported);
  const auto Queue = getQueue();
  const auto Kernel = getKernel(Queue);
  const auto maxWorkGroupSize = Kernel.template ext_oneapi_get_info<
    syclex::info::kernel_queue_specific::max_work_group_size>(Queue);
  ASSERT_EQ(maxWorkGroupSize, 0); 
}

TEST(LaunchQueries, GetWorkGroupSizeExceptionCode)
{
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_replace_callback("urKernelGetGroupInfo",
                                          &redefine_urKernelGetGroupInfo_Exception);
  const auto Queue = getQueue();
  const auto Kernel = getKernel(Queue);
  EXPECT_THROW(Kernel.template ext_oneapi_get_info<
    syclex::info::kernel_queue_specific::max_work_group_size>(Queue), sycl::exception);
}

TEST(LaunchQueries, GetMaxWorkGroupItemSizes3DSuccess)
{
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_replace_callback("urKernelGetGroupInfo",
                                          &redefine_urKernelGetGroupInfo_Success);
  const auto Queue = getQueue();
  const auto Kernel = getKernel(Queue);
  const auto maxWorkGroupItemSizes = Kernel.template ext_oneapi_get_info<
    syclex::info::kernel_queue_specific::max_work_item_sizes<3>>(Queue);
  const auto result = std::is_same_v<std::remove_cv_t<decltype(maxWorkGroupItemSizes)>, sycl::id<3>>;
  ASSERT_TRUE(result);
  ASSERT_EQ(decltype(maxWorkGroupItemSizes)::dimensions, 3);
  ASSERT_EQ(maxWorkGroupItemSizes[0], 123);
  ASSERT_EQ(maxWorkGroupItemSizes[1], 213);
  ASSERT_EQ(maxWorkGroupItemSizes[2], 321);
}

TEST(LaunchQueries, GetMaxWorkGroupItemSizes2DSuccess)
{
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_replace_callback("urKernelGetGroupInfo",
                                          &redefine_urKernelGetGroupInfo_Success);
  const auto Queue = getQueue();
  const auto Kernel = getKernel(Queue);
  const auto maxWorkGroupItemSizes = Kernel.template ext_oneapi_get_info<
    syclex::info::kernel_queue_specific::max_work_item_sizes<2>>(Queue);
  const auto result = std::is_same_v<std::remove_cv_t<decltype(maxWorkGroupItemSizes)>, sycl::id<2>>;
  ASSERT_TRUE(result);
  ASSERT_EQ(decltype(maxWorkGroupItemSizes)::dimensions, 2);
  ASSERT_EQ(maxWorkGroupItemSizes[0], 123);
  ASSERT_EQ(maxWorkGroupItemSizes[1], 213);
}

TEST(LaunchQueries, GetMaxWorkGroupItemSizes1DSuccess)
{
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_replace_callback("urKernelGetGroupInfo",
                                          &redefine_urKernelGetGroupInfo_Success);
  const auto Queue = getQueue();
  const auto Kernel = getKernel(Queue);
  const auto maxWorkGroupItemSizes = Kernel.template ext_oneapi_get_info<
    syclex::info::kernel_queue_specific::max_work_item_sizes<1>>(Queue);
  const auto result = std::is_same_v<std::remove_cv_t<decltype(maxWorkGroupItemSizes)>, sycl::id<1>>;
  ASSERT_TRUE(result);
  ASSERT_EQ(decltype(maxWorkGroupItemSizes)::dimensions, 1);
  ASSERT_EQ(maxWorkGroupItemSizes[0], 123);
}

TEST(LaunchQueries, GetMaxWorkGroupItemSizesUnsupported)
{
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_replace_callback("urKernelGetGroupInfo",
                                          &redefine_urKernelGetGroupInfo_Unsupported);
  const auto Queue = getQueue();
  const auto Kernel = getKernel(Queue);
  const auto maxWorkGroupItemSizes = Kernel.template ext_oneapi_get_info<
    syclex::info::kernel_queue_specific::max_work_item_sizes<3>>(Queue);
  ASSERT_EQ(maxWorkGroupItemSizes[0], 0);
  ASSERT_EQ(maxWorkGroupItemSizes[1], 0);
  ASSERT_EQ(maxWorkGroupItemSizes[2], 0); 
}

TEST(LaunchQueries, GetMaxWorkGroupItemSizesExceptionCode)
{
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_replace_callback("urKernelGetGroupInfo",
                                          &redefine_urKernelGetGroupInfo_Exception);
  const auto Queue = getQueue();
  const auto Kernel = getKernel(Queue);
  EXPECT_THROW(Kernel.template ext_oneapi_get_info<
    syclex::info::kernel_queue_specific::max_work_item_sizes<3>>(Queue), sycl::exception);
}

TEST(LaunchQueries, GetMaxSubGroupSize3DSuccess)
{
  sycl::unittest::UrMock<> Mock;
  const auto Queue = getQueue();
  const auto Kernel = getKernel(Queue);
  const auto maxSubGroupSize = Kernel.template ext_oneapi_get_info<
    syclex::info::kernel_queue_specific::max_sub_group_size>(Queue, sycl::range<3>{1,1,1});
  const auto result = std::is_same_v<std::remove_cv_t<decltype(maxSubGroupSize)>, uint32_t>;
  ASSERT_TRUE(result);
  ASSERT_EQ(maxSubGroupSize, 123);
}

TEST(LaunchQueries, GetMaxSubGroupSize2DSuccess)
{
  sycl::unittest::UrMock<> Mock;
  const auto Queue = getQueue();
  const auto Kernel = getKernel(Queue);
  const auto maxSubGroupSize = Kernel.template ext_oneapi_get_info<
    syclex::info::kernel_queue_specific::max_sub_group_size>(Queue, sycl::range<2>{1,1});
  const auto result = std::is_same_v<std::remove_cv_t<decltype(maxSubGroupSize)>, uint32_t>;
  ASSERT_TRUE(result);
  ASSERT_EQ(maxSubGroupSize, 123);
}

TEST(LaunchQueries, GetMaxSubGroupSize1DSuccess)
{
  sycl::unittest::UrMock<> Mock;
  const auto Queue = getQueue();
  const auto Kernel = getKernel(Queue);
  const auto maxSubGroupSize = Kernel.template ext_oneapi_get_info<
    syclex::info::kernel_queue_specific::max_sub_group_size>(Queue, sycl::range<1>{1});
  const auto result = std::is_same_v<std::remove_cv_t<decltype(maxSubGroupSize)>, uint32_t>;
  ASSERT_TRUE(result);
  ASSERT_EQ(maxSubGroupSize, 123);
}

TEST(LaunchQueries, GetMaxSubGroupSizeUnsupported)
{
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_replace_callback("urKernelGetSubGroupInfo",
                                          &redefine_urKernelGetGroupInfo_Unsupported);
  const auto Queue = getQueue();
  const auto Kernel = getKernel(Queue);
  const auto maxSubGroupSize = Kernel.template ext_oneapi_get_info<
    syclex::info::kernel_queue_specific::max_sub_group_size>(Queue, sycl::range<3>{1,1,1});
  ASSERT_EQ(maxSubGroupSize, 0);
}

TEST(LaunchQueries, GetMaxSubGroupSizeExceptionCode)
{
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_replace_callback("urKernelGetSubGroupInfo",
                                          &redefine_urKernelGetGroupInfo_Exception);
  const auto Queue = getQueue();
  const auto Kernel = getKernel(Queue);
  EXPECT_THROW(Kernel.template ext_oneapi_get_info<
    syclex::info::kernel_queue_specific::max_sub_group_size>(Queue, sycl::range<3>{1, 1, 1});, sycl::exception);
  EXPECT_THROW(Kernel.template ext_oneapi_get_info<
    syclex::info::kernel_queue_specific::max_sub_group_size>(Queue, sycl::range<2>{1, 1});, sycl::exception);
  EXPECT_THROW(Kernel.template ext_oneapi_get_info<
    syclex::info::kernel_queue_specific::max_sub_group_size>(Queue, sycl::range<1>{1});, sycl::exception);
}

TEST(LaunchQueries, GetMaxSubGroupSizeExceptionSize)
{
  sycl::unittest::UrMock<> Mock;
  const auto Queue = getQueue();
  const auto Kernel = getKernel(Queue);
  EXPECT_THROW(Kernel.template ext_oneapi_get_info<
    syclex::info::kernel_queue_specific::max_sub_group_size>(Queue, sycl::range<3>{0, 0, 0});, sycl::exception);
  EXPECT_THROW(Kernel.template ext_oneapi_get_info<
    syclex::info::kernel_queue_specific::max_sub_group_size>(Queue, sycl::range<2>{0, 0});, sycl::exception);
  EXPECT_THROW(Kernel.template ext_oneapi_get_info<
    syclex::info::kernel_queue_specific::max_sub_group_size>(Queue, sycl::range<1>{0});, sycl::exception);
}

TEST(LaunchQueries, GetNumSubGroups3DSuccess)
{
  sycl::unittest::UrMock<> Mock;
  const auto Queue = getQueue();
  const auto Kernel = getKernel(Queue);
  const auto NumSubGroups = Kernel.template ext_oneapi_get_info<
    syclex::info::kernel_queue_specific::num_sub_groups>(Queue, sycl::range<3>{1, 1, 1});
  const auto result = std::is_same_v<std::remove_cv_t<decltype(NumSubGroups)>, uint32_t>;
  ASSERT_TRUE(result);
  ASSERT_EQ(NumSubGroups, 123);
}

TEST(LaunchQueries, GetNumSubGroups2DSuccess)
{
  sycl::unittest::UrMock<> Mock;
  const auto Queue = getQueue();
  const auto Kernel = getKernel(Queue);
  const auto NumSubGroups = Kernel.template ext_oneapi_get_info<
    syclex::info::kernel_queue_specific::num_sub_groups>(Queue, sycl::range<2>{1, 1});
  const auto result = std::is_same_v<std::remove_cv_t<decltype(NumSubGroups)>, uint32_t>;
  ASSERT_TRUE(result);
  ASSERT_EQ(NumSubGroups, 123);
}

TEST(LaunchQueries, GetNumSubGroups1DSuccess)
{
  sycl::unittest::UrMock<> Mock;
  const auto Queue = getQueue();
  const auto Kernel = getKernel(Queue);
  const auto NumSubGroups = Kernel.template ext_oneapi_get_info<
    syclex::info::kernel_queue_specific::num_sub_groups>(Queue, sycl::range<1>{1});
  const auto result = std::is_same_v<std::remove_cv_t<decltype(NumSubGroups)>, uint32_t>;
  ASSERT_TRUE(result);
  ASSERT_EQ(NumSubGroups, 123);
}

TEST(LaunchQueries, GetNumSubGroupsUnsupported)
{
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_replace_callback("urKernelGetSubGroupInfo",
                                          &redefine_urKernelGetGroupInfo_Unsupported);
  const auto Queue = getQueue();
  const auto Kernel = getKernel(Queue);
  const auto NumSubGroups = Kernel.template ext_oneapi_get_info<
    syclex::info::kernel_queue_specific::num_sub_groups>(Queue, sycl::range<3>{1, 1, 1});
  ASSERT_EQ(NumSubGroups, 0);
}

TEST(LaunchQueries, GetNumSubGroupsExceptionCode)
{
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_replace_callback("urKernelGetSubGroupInfo",
                                          &redefine_urKernelGetGroupInfo_Exception);
  const auto Queue = getQueue();
  const auto Kernel = getKernel(Queue);
  EXPECT_THROW(Kernel.template ext_oneapi_get_info<
    syclex::info::kernel_queue_specific::num_sub_groups>(Queue, sycl::range<3>{1, 1, 1});, sycl::exception);
  EXPECT_THROW(Kernel.template ext_oneapi_get_info<
    syclex::info::kernel_queue_specific::num_sub_groups>(Queue, sycl::range<2>{1, 1});, sycl::exception);
  EXPECT_THROW(Kernel.template ext_oneapi_get_info<
    syclex::info::kernel_queue_specific::num_sub_groups>(Queue, sycl::range<1>{1});, sycl::exception);
}

TEST(LaunchQueries, GetNumSubGroupsExceptionSize)
{
  sycl::unittest::UrMock<> Mock;
  const auto Queue = getQueue();
  const auto Kernel = getKernel(Queue);
  EXPECT_THROW(Kernel.template ext_oneapi_get_info<
    syclex::info::kernel_queue_specific::num_sub_groups>(Queue, sycl::range<3>{0, 0, 0});, sycl::exception);
  EXPECT_THROW(Kernel.template ext_oneapi_get_info<
    syclex::info::kernel_queue_specific::num_sub_groups>(Queue, sycl::range<2>{0, 0});, sycl::exception);
  EXPECT_THROW(Kernel.template ext_oneapi_get_info<
    syclex::info::kernel_queue_specific::num_sub_groups>(Queue, sycl::range<1>{0});, sycl::exception);
}
