// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests that marray works in sub-group shuffles.

#include <sycl/detail/core.hpp>
#include <sycl/group_algorithm.hpp>
#include <sycl/marray.hpp>

static constexpr size_t NumElems = 5;
static constexpr size_t NumWorkItems = 64;

using MarrayT = sycl::marray<int, NumElems>;

std::string MarrayToString(const MarrayT &Marr) {
  std::string S = "{";
  for (size_t I = 0; I < Marr.size(); ++I) {
    S += std::to_string(Marr[I]);
    if (I != Marr.size() - 1)
      S += ",";
  }
  return S + "}";
}

int CheckMarray(const MarrayT &Lhs, const MarrayT &Rhs) {
  auto Comp = Lhs == Rhs;
  if (!std::all_of(Comp.begin(), Comp.end(), [](bool B) { return B; })) {
    std::cout << "Failed: " << MarrayToString(Lhs)
              << " != " << MarrayToString(Rhs) << std::endl;
    return 1;
  }
  return 0;
}

int main() {
  sycl::queue Q;

  MarrayT ShiftLeftRes[NumWorkItems];
  MarrayT ShiftRightRes[NumWorkItems];
  MarrayT PermuteXorRes[NumWorkItems];
  MarrayT SelectRes[NumWorkItems];
  unsigned SubGroupSize = 0;

  {
    sycl::buffer<MarrayT, 1> ShiftLeftResBuff{ShiftLeftRes, NumWorkItems};
    sycl::buffer<MarrayT, 1> ShiftRightResBuff{ShiftRightRes, NumWorkItems};
    sycl::buffer<MarrayT, 1> PermuteXorResBuff{PermuteXorRes, NumWorkItems};
    sycl::buffer<MarrayT, 1> SelectResBuff{SelectRes, NumWorkItems};
    sycl::buffer<unsigned, 1> SubGroupSizeBuff{&SubGroupSize, 1};

    Q.submit([&](sycl::handler &CGH) {
      sycl::accessor ShiftLeftResAcc{ShiftLeftResBuff, CGH, sycl::write_only};
      sycl::accessor ShiftRightResAcc{ShiftRightResBuff, CGH, sycl::write_only};
      sycl::accessor PermuteXorResAcc{PermuteXorResBuff, CGH, sycl::write_only};
      sycl::accessor SelectResAcc{SelectResBuff, CGH, sycl::write_only};
      sycl::accessor SubGroupSizeAcc{SubGroupSizeBuff, CGH, sycl::write_only};

      CGH.parallel_for(
          sycl::nd_range<1>{sycl::range<1>{NumWorkItems},
                            sycl::range<1>{NumWorkItems}},
          [=](sycl::nd_item<1> It) {
            int GID = It.get_global_linear_id();
            int ValueOffset = GID * NumElems;
            MarrayT ItemVal{0};
            for (int I = 0; I < NumElems; ++I)
              ItemVal[I] = ValueOffset + I;

            sycl::sub_group SG = It.get_sub_group();
            if (GID == 0)
              SubGroupSizeAcc[0] = SG.get_local_linear_range();

            ShiftLeftResAcc[GID] = sycl::shift_group_left(SG, ItemVal);
            ShiftRightResAcc[GID] = sycl::shift_group_right(SG, ItemVal);
            PermuteXorResAcc[GID] = sycl::permute_group_by_xor(SG, ItemVal, 1);
            SelectResAcc[GID] = sycl::select_from_group(SG, ItemVal, 0);
          });
    });
  }

  int Failures = 0;

  for (size_t SGI = 0; SGI < NumWorkItems / SubGroupSize; ++SGI) {
    std::cout << "Checking results for sub-group " << SGI << std::endl;

    size_t GIDOffset = SGI * SubGroupSize;

    std::cout << "Checking results for shuffle-left" << std::endl;
    // For left shift the value of the last item is undefined, as it goes
    // outside the range.
    for (size_t LID = 0; LID < SubGroupSize - 1; ++LID) {
      size_t GID = GIDOffset + LID;
      MarrayT Expected{0};
      for (int I = 0; I < NumElems; ++I)
        Expected[I] = (GID + 1) * NumElems + I;
      Failures += CheckMarray(ShiftLeftRes[GID], Expected);
    }

    std::cout << "Checking results for shuffle-right" << std::endl;
    // For right shift the value of the first item is undefined, as it goes
    // outside the range, so we start with the second item.
    for (size_t LID = 1; LID < SubGroupSize; ++LID) {
      size_t GID = GIDOffset + LID;
      MarrayT Expected{0};
      for (int I = 0; I < NumElems; ++I)
        Expected[I] = (GID - 1) * NumElems + I;
      Failures += CheckMarray(ShiftRightRes[GID], Expected);
    }

    std::cout << "Checking results for shuffle-xor" << std::endl;
    for (size_t LID = 0; LID < SubGroupSize; ++LID) {
      size_t GID = GIDOffset + LID;
      MarrayT Expected{0};
      for (int I = 0; I < NumElems; ++I)
        Expected[I] = (GIDOffset + (LID ^ 1)) * NumElems + I;
      Failures += CheckMarray(PermuteXorRes[GID], Expected);
    }

    std::cout << "Checking results for shuffle-select" << std::endl;
    MarrayT SelectExpected{0};
    for (int I = 0; I < NumElems; ++I)
      SelectExpected[I] = GIDOffset * NumElems + I;
    for (size_t LID = 0; LID < SubGroupSize; ++LID)
      Failures += CheckMarray(SelectRes[GIDOffset + LID], SelectExpected);
  }

  return Failures;
}
