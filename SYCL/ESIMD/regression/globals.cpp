// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// TODO: esimd_emulator fails due to unimplemented sub-group support
// XFAIL: esimd_emulator
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// This test checks loads from SPIRV builtin globals work correctly in ESIMD.

#include <cstdint>

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

template <int Dim> id<Dim> asId(range<Dim> V) {
  if constexpr (Dim == 1)
    return id<1>{V[0]};
  if constexpr (Dim == 2)
    return id<2>{V[1], V[0]};
  if constexpr (Dim == 3)
    return id<3>{V[2], V[1], V[0]};
}

template <int Dim>
std::ostream &operator<<(std::ostream &OS, const id<Dim> &V) {
  OS << "{";
  if constexpr (Dim >= 3)
    OS << V[2] << ", ";
  if constexpr (Dim >= 2)
    OS << V[1] << ", ";
  OS << V[0] << "}";
  return OS;
}

template <int Dim>
std::ostream &operator<<(std::ostream &OS, const range<Dim> &V) {
  return OS << asId(V);
}

// Returns true if the test passed, false otherwise.
template <typename T1, typename T2>
bool check(std::string VarName, T1 Expected, T2 Computed, size_t I) {
  if (Computed != Expected) {
    std::cerr << "Error: " << VarName << "[" << I << "]: expected: {"
              << Expected << ", computed: " << Computed << std::endl;
    return false;
  }
  return true;
}

template <int Dim> id<Dim> zeroId() {
  if constexpr (Dim == 1)
    return id<Dim>{0};
  if constexpr (Dim == 2)
    return id<Dim>{0, 0};
  if constexpr (Dim == 3)
    return id<Dim>{0, 0, 0};
}

// Increases 'Id' if it does not exceed the range 'R'
template <int Dim> void advanceId(id<Dim> &Id, range<Dim> R) {
  if constexpr (Dim == 3) {
    if (Id[2] + 1 < R[2]) {
      Id[2]++;
      return;
    }
    Id[2] = 0;
  }
  if constexpr (Dim >= 2) {
    if (Id[1] + 1 < R[1]) {
      Id[1]++;
      return;
    }
    Id[1] = 0;
  }
  if (Id[0] + 1 < R[0])
    Id[0]++;
}

template <int Dim> int testXD(queue &Q, nd_range<Dim> NDR) {
  device Dev = Q.get_device();
  context Context = Q.get_context();

  std::cout << "Start test case for range: {" << NDR.get_global_range() << ", "
            << NDR.get_local_range() << "}" << std::endl;

  size_t GlobalRange = NDR.get_global_range().size();
  size_t LocalRange = NDR.get_local_range().size();

  id<Dim> *GlobalId = sycl::malloc_shared<id<Dim>>(GlobalRange, Dev, Context);
  range<Dim> *GlobalSize =
      sycl::malloc_shared<range<Dim>>(GlobalRange, Dev, Context);

  id<Dim> *LocalId = sycl::malloc_shared<id<Dim>>(GlobalRange, Dev, Context);
  range<Dim> *LocalSize =
      sycl::malloc_shared<range<Dim>>(GlobalRange, Dev, Context);

  uint32_t *SubGroupLocalId =
      sycl::malloc_shared<uint32_t>(GlobalRange, Dev, Context);
  uint32_t *SubGroupSize =
      sycl::malloc_shared<uint32_t>(GlobalRange, Dev, Context);
  uint32_t *SubGroupMaxSize =
      sycl::malloc_shared<uint32_t>(GlobalRange, Dev, Context);

  id<Dim> *GroupId = sycl::malloc_shared<id<Dim>>(GlobalRange, Dev, Context);
  range<Dim> *NumGroups =
      sycl::malloc_shared<range<Dim>>(GlobalRange, Dev, Context);

  Q.submit([&](sycl::handler &CGH) {
     namespace seoe = sycl::ext::oneapi::experimental;
     CGH.parallel_for(NDR, [=](sycl::nd_item<Dim> NdId) SYCL_ESIMD_KERNEL {
       size_t I = NdId.get_global_linear_id();
       id<Dim> Id = NdId.get_global_id();

       GlobalId[I] = Id;
       GlobalSize[I] = NdId.get_global_range();

       LocalId[I] = NdId.get_local_id();
       LocalSize[I] = NdId.get_local_range();

       SubGroupLocalId[I] = seoe::this_sub_group().get_local_id();
       SubGroupSize[I] = seoe::this_sub_group().get_local_range()[0];
       SubGroupMaxSize[I] = seoe::this_sub_group().get_max_local_range()[0];

       GroupId[I] = NdId.get_group().get_group_id();
       NumGroups[I] = NdId.get_group_range();
     });
   }).wait();

  id<Dim> Id = zeroId<Dim>();
  int Error = 0;
  for (size_t I = 0; I < NDR.get_global_range().size(); I++) {
    if (!check("GlobalId", Id, GlobalId[I], I) ||
        !check("GlobalSize", NDR.get_global_range(), GlobalSize[I], I) ||
        !check("LocalId", Id % NDR.get_local_range(), LocalId[I], I) ||
        !check("LocalSize", NDR.get_local_range(), LocalSize[I], I) ||
        !check("SubGroupLocalId", 0, SubGroupLocalId[I], I) ||
        !check("SubGroupSize", 1, SubGroupSize[I], I) ||
        !check("SubGroupMaxSize", 1, SubGroupMaxSize[I], I) ||
        !check("GroupId", Id / NDR.get_local_range(), GroupId[I], I) ||
        !check("NumGroups", NDR.get_group_range(), NumGroups[I], I)) {
      Error = 1;
      break;
    }
    advanceId(Id, NDR.get_global_range());
  }

  free(GlobalId, Q);
  free(GlobalSize, Q);
  free(LocalId, Q);
  free(LocalSize, Q);
  free(SubGroupLocalId, Q);
  free(SubGroupSize, Q);
  free(SubGroupMaxSize, Q);
  free(GroupId, Q);
  free(NumGroups, Q);
  return Error;
}

int main() {
  queue Q;

  int NumErrors = 0;
  NumErrors += testXD(Q, nd_range<1>{range<1>{10}, range<1>{5}});
  NumErrors += testXD(Q, nd_range<2>{range<2>{8, 9}, range<2>{2, 3}});
  NumErrors += testXD(Q, nd_range<3>{range<3>{6, 12, 17}, range<3>{2, 3, 1}});

  if (NumErrors)
    std::cerr << "Test FAILED" << std::endl;
  else
    std::cout << "Test passed" << std::endl;

  return NumErrors;
}
