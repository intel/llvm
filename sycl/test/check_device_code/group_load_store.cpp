// DEFINE: %{fcflags} = --check-prefixes %if windows %{CHECK,CHECK-WIN%} %else %{CHECK,CHECK-LIN%}

// TODO: Remove -opaque-pointers when they are on by default.
// We want to use them right away for two reasons:
//  1) Less maintenance during the future switch
//  2) Generated LLVM IR is nicer and more readable in this mode
// RUN: %clangxx -fsycl-device-only -S -emit-llvm -Xclang -opaque-pointers -fno-sycl-instrument-device-code  -o - %s | FileCheck %s %{fcflags}
#include <sycl/sycl.hpp>

using namespace sycl;

using namespace sycl::ext::oneapi::experimental;

using empty_props = decltype(properties());
using full_sg_props =
    decltype(properties(sycl::ext::oneapi::experimental::property::full_sg));

template <typename T>
using plain_global_ptr = typename sycl::detail::DecoratedType<
    T, access::address_space::global_space>::type *;

template <typename T>
using plain_local_ptr = typename sycl::detail::DecoratedType<
    T, access::address_space::local_space>::type *;

template SYCL_EXTERNAL void sycl::ext::oneapi::experimental::group_load<
    sycl::sub_group, plain_global_ptr<int>, int, full_sg_props>(
    sycl::sub_group, plain_global_ptr<int>, int &, full_sg_props);
// CHECK-LABEL: define {{.*}}group_load
// CHECK-NEXT: entry:
// CHECK-NEXT:   call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 noundef 3, i32 noundef 3, i32 noundef 912)
// CHECK-NEXT:   [[BLOCK_LOAD:%.*]] = tail call spir_func noundef i32 @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(ptr addrspace(1) noundef [[IN_PTR:%.*]])
// CHECK-NEXT:   store i32 [[BLOCK_LOAD]], ptr addrspace(4) [[OUT:%.*]]
// CHECK-NEXT:   ret void

template SYCL_EXTERNAL void sycl::ext::oneapi::experimental::group_load<
    sycl::sub_group, global_ptr<int, access::decorated::yes>, int,
    full_sg_props>(sycl::sub_group, global_ptr<int, access::decorated::yes>,
                   int &, full_sg_props);
// CHECK-LABEL: define {{.*}}group_load
// CHECK-NEXT: entry:
// sycl::multi_ptr is passed via "ptr noundef
// byval(%"class.sycl::_V1::multi_ptr") align 8 %in_ptr"
// CHECK-NEXT:   [[IN_PTR_LOAD:%.*]] = load i64
// CHECK-NEXT:   [[IN_PTR:%.*]] = inttoptr i64 [[IN_PTR_LOAD]] to ptr addrspace(1)
// CHECK-NEXT:   call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 noundef 3, i32 noundef 3, i32 noundef 912)
// CHECK-NEXT:   [[BLOCK_LOAD:%.*]] = tail call spir_func noundef i32 @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(ptr addrspace(1) noundef [[IN_PTR]])
// CHECK-NEXT:   store i32 [[BLOCK_LOAD]], ptr addrspace(4) [[OUT:%.*]]
// CHECK-NEXT:   ret void

template SYCL_EXTERNAL void sycl::ext::oneapi::experimental::group_load<
    sycl::sub_group, plain_local_ptr<int>, int, full_sg_props>(
    sycl::sub_group, plain_local_ptr<int>, int &, full_sg_props);
// CHECK-LABEL: define {{.*}}group_load
// CHECK-NEXT: entry:
// CHECK-NEXT:   call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 noundef 3, i32 noundef 3, i32 noundef 912)
// CHECK-NEXT:   [[SG_LID:%.*]] = load i32, ptr addrspace(1) @__spirv_BuiltInSubgroupLocalInvocationId
// CHECK-NEXT:   [[SG_LID_64:%.*]] = sext i32 [[SG_LID]] to i64
// CHECK-NEXT:   [[GEP:%.*]] = getelementptr inbounds i32, ptr addrspace(3) [[IN_PTR:%.*]], i64 [[SG_LID_64]]
// CHECK-NEXT:   [[LD:%.*]] = load i32, ptr addrspace(3) [[GEP]]
// CHECK-NEXT:   store i32 [[LD]], ptr addrspace(4) [[OUT:%.*]]
// CHECK-NEXT:   call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 noundef 3, i32 noundef 3, i32 noundef 912)
// CHECK-NEXT:   ret void

template SYCL_EXTERNAL void sycl::ext::oneapi::experimental::group_load<
    sycl::sub_group, int *, int, full_sg_props>(sycl::sub_group, int *, int &,
                                                full_sg_props);
// CHECK-LABEL: define {{.*}}group_load
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[TRY_CAST:%.*]] = tail call spir_func noundef ptr addrspace(1) @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi(ptr addrspace(4) noundef [[IN_PTR:%.*]], i32 noundef 5)
// CHECK-NEXT:   [[COND:%.*]] = icmp eq ptr addrspace(1) [[TRY_CAST]], null
// CHECK-NEXT:   br i1 [[COND]], label %[[GENERIC:.*]], label %[[GLOBAL:.*]]
// CHECK-EMPTY:
// CHECK-NEXT: [[GLOBAL]]:
// CHECK-NEXT:   call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 noundef 3, i32 noundef 3, i32 noundef 912)
// CHECK-NEXT:   [[BLOCK_LOAD:%.*]] =  tail call spir_func noundef i32 @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(ptr addrspace(1) noundef nonnull [[TRY_CAST]])
// CHECK-NEXT:   store i32 [[BLOCK_LOAD]], ptr addrspace(4) [[OUT:%.*]]
// CHECK-NEXT:   br label %[[RET:.*]]
// CHECK-EMPTY:
// CHECK-NEXT: [[GENERIC]]:
// CHECK-NEXT:   call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 noundef 3, i32 noundef 3, i32 noundef 912)
// CHECK-NEXT:   [[SG_LID:%.*]] = load i32, ptr addrspace(1) @__spirv_BuiltInSubgroupLocalInvocationId
// CHECK-NEXT:   [[SG_LID_64:%.*]] = sext i32 [[SG_LID]] to i64
// CHECK-NEXT:   [[GEP:%.*]] = getelementptr inbounds i32, ptr addrspace(4) [[IN_PTR]], i64 [[SG_LID_64]]
// CHECK-NEXT:   [[LD:%.*]] = load i32, ptr addrspace(4) [[GEP]]
// CHECK-NEXT:   store i32 [[LD]], ptr addrspace(4) [[OUT]]
// CHECK-NEXT:   call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 noundef 3, i32 noundef 3, i32 noundef 912)
// CHECK-NEXT:   br label %[[RET]]
// CHECK-EMPTY:
// CHECK-NEXT: [[RET]]:
// CHECK-NEXT:   ret void

using full_sg_striped_props = decltype(properties(
    sycl::ext::oneapi::experimental::property::full_sg,
    sycl::ext::oneapi::experimental::property::data_placement<
        group_algorithm_data_placement::striped>));
template SYCL_EXTERNAL void sycl::ext::oneapi::experimental::group_load<
    sycl::sub_group, plain_global_ptr<long long>, long long, 2,
    full_sg_striped_props>(sycl::sub_group, plain_global_ptr<long long>,
                           vec<long long, 2> &, full_sg_striped_props);
// CHECK-LABEL: define {{.*}}group_load
// CHECK-NEXT: entry:
// CHECK-NEXT:   call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 noundef 3, i32 noundef 3, i32 noundef 912)
// CHECK-LIN-NEXT:   [[BLOCK_LOAD:%.*]] = tail call spir_func noundef <2 x i64> @_Z30__spirv_SubgroupBlockReadINTELIDv2_mET_PU3AS1Km(ptr addrspace(1) noundef [[IN_PTR:%.*]])
// CHECK-WIN-NEXT:   [[BLOCK_LOAD:%.*]] = tail call spir_func noundef <2 x i64> @_Z30__spirv_SubgroupBlockReadINTELIDv2_yET_PU3AS1Ky(ptr addrspace(1) noundef [[IN_PTR:%.*]])
// CHECK-NEXT:   [[EXTRACT_0:%.*]] = extractelement <2 x i64> [[BLOCK_LOAD]], i64 0
// CHECK-NEXT:   store i64 [[EXTRACT_0]], ptr addrspace(4) [[OUT_PTR:%[^,]*]]
// CHECK-NEXT:   [[EXTRACT_1:%.*]] = extractelement <2 x i64> [[BLOCK_LOAD]], i64 1
// CHECK-NEXT:   [[GEP:%.*]] = getelementptr inbounds i64, ptr addrspace(4) [[OUT_PTR]], i64 1
// CHECK-NEXT:   store i64 [[EXTRACT_1]], ptr addrspace(4) [[GEP]]
// CHECK-NEXT:   ret void

template SYCL_EXTERNAL void sycl::ext::oneapi::experimental::group_load<
    sycl::sub_group, plain_global_ptr<int>, int, 16, full_sg_striped_props>(
    sycl::sub_group, plain_global_ptr<int>, vec<int, 16> &,
    full_sg_striped_props);
// CHECK-LABEL: define {{.*}}group_load
// CHECK-NEXT: entry:
// CHECK-NEXT:   call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 noundef 3, i32 noundef 3, i32 noundef 912)
// CHECK-NEXT:   [[SG_SIZE:%.*]] = load i32, ptr addrspace(1) @__spirv_BuiltInSubgroupMaxSize
// CHECK-NEXT:   [[SG_SIZE_64:%.*]] = zext i32 [[SG_SIZE]] to i64
// CHECK-NEXT:   [[BYTES_PER_SG:%.*]] = shl nuw nsw i64 [[SG_SIZE_64]], 3
// CHECK-NEXT:   [[BLOCK_LOAD_0:%.*]] = tail call spir_func noundef <8 x i32> @_Z30__spirv_SubgroupBlockReadINTELIDv8_jET_PU3AS1Kj(ptr addrspace(1) noundef [[IN_PTR:%.*]])
// CHECK-NEXT:   [[GEP:%.*]] = getelementptr inbounds i32, ptr addrspace(1) [[IN_PTR]], i64 [[BYTES_PER_SG]]
// CHECK-NEXT:   [[BLOCK_LOAD_1:%.*]] = tail call spir_func noundef <8 x i32> @_Z30__spirv_SubgroupBlockReadINTELIDv8_jET_PU3AS1Kj(ptr addrspace(1) noundef [[GEP]])
// 16 * (extract/gep/store) - 1 (0-th gep)
// CHECK-COUNT-47: {{extractelement|getelementptr|store i32}}
// CHECK-NEXT:   ret void
