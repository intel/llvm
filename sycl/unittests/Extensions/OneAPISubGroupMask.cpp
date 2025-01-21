#include <gtest/gtest.h>

#include <sycl/sycl.hpp>

#include <algorithm>

class SubGroupMask : public testing::TestWithParam<size_t> {
protected:
  sycl::ext::oneapi::sub_group_mask MaskZero =
      sycl::detail::Builder::createSubGroupMask<
          sycl::ext::oneapi::sub_group_mask>(0, GetParam());
  sycl::ext::oneapi::sub_group_mask MaskOne =
      sycl::detail::Builder::createSubGroupMask<
          sycl::ext::oneapi::sub_group_mask>(1, GetParam());
  sycl::ext::oneapi::sub_group_mask MaskAllOnes =
      sycl::detail::Builder::createSubGroupMask<
          sycl::ext::oneapi::sub_group_mask>(
          static_cast<sycl::ext::oneapi::sub_group_mask::BitsType>(-1),
          GetParam());
};

TEST_P(SubGroupMask, SubscriptOperatorBool) {
  const auto MZ = MaskZero;
  const auto MO = MaskOne;
  const auto MAO = MaskAllOnes;
  ASSERT_FALSE(MZ[0]);
  ASSERT_TRUE(MO[0]);
  ASSERT_TRUE(MAO[0]);

  for (size_t I = 1; I < GetParam(); ++I) {
    ASSERT_FALSE(MZ[I]);
    ASSERT_FALSE(MO[I]);
    ASSERT_TRUE(MAO[I]);
  }
}

TEST_P(SubGroupMask, SubscriptOperator) {
  ASSERT_FALSE(MaskZero[0]);
  ASSERT_TRUE(MaskOne[0]);
  ASSERT_TRUE(MaskAllOnes[0]);

  for (size_t I = 1; I < GetParam(); ++I) {
    ASSERT_FALSE(MaskZero[I]);
    ASSERT_FALSE(MaskOne[I]);
    ASSERT_TRUE(MaskAllOnes[I]);
  }
}

TEST_P(SubGroupMask, Test) {
  ASSERT_FALSE(MaskZero.test(0));
  ASSERT_TRUE(MaskOne.test(0));
  ASSERT_TRUE(MaskAllOnes.test(0));

  for (size_t I = 1; I < GetParam(); ++I) {
    ASSERT_FALSE(MaskZero.test(I));
    ASSERT_FALSE(MaskOne.test(I));
    ASSERT_TRUE(MaskAllOnes.test(I));
  }
}

TEST_P(SubGroupMask, All) {
  ASSERT_FALSE(MaskZero.all());
  ASSERT_FALSE(MaskOne.all());
  ASSERT_TRUE(MaskAllOnes.all());
}

TEST_P(SubGroupMask, Any) {
  ASSERT_FALSE(MaskZero.any());
  ASSERT_TRUE(MaskOne.any());
  ASSERT_TRUE(MaskAllOnes.any());
}

TEST_P(SubGroupMask, None) {
  ASSERT_TRUE(MaskZero.none());
  ASSERT_FALSE(MaskOne.none());
  ASSERT_FALSE(MaskAllOnes.none());
}

TEST_P(SubGroupMask, Count) {
  ASSERT_EQ(MaskZero.count(), 0u);
  ASSERT_EQ(MaskOne.count(), 1u);
  ASSERT_EQ(MaskAllOnes.count(), GetParam());
}

TEST_P(SubGroupMask, Size) {
  ASSERT_EQ(MaskZero.size(), GetParam());
  ASSERT_EQ(MaskOne.size(), GetParam());
  ASSERT_EQ(MaskAllOnes.size(), GetParam());
}

TEST_P(SubGroupMask, SetAll) {
  MaskZero.set();
  ASSERT_TRUE(MaskZero.all());
  MaskOne.set();
  ASSERT_TRUE(MaskOne.all());
}

TEST_P(SubGroupMask, Set) {
  MaskZero.set(GetParam() / 2);
  ASSERT_TRUE(MaskZero[GetParam() / 2]);
  MaskOne.set(0, false);
  ASSERT_FALSE(MaskOne[0]);
}

TEST_P(SubGroupMask, ResetAll) {
  MaskOne.reset();
  ASSERT_TRUE(MaskOne.none());
  MaskAllOnes.reset();
  ASSERT_TRUE(MaskAllOnes.none());
}

TEST_P(SubGroupMask, Reset) {
  MaskOne.reset(0);
  ASSERT_FALSE(MaskOne[0]);
  MaskAllOnes.reset(GetParam() / 2);
  ASSERT_FALSE(MaskAllOnes[GetParam() / 2]);
}

TEST_P(SubGroupMask, FlipAll) {
  MaskZero.flip();
  ASSERT_TRUE(MaskZero.all());
  MaskAllOnes.flip();
  ASSERT_TRUE(MaskAllOnes.none());
  MaskOne.flip();
  ASSERT_FALSE(MaskOne[0]);
  for (size_t I = 1; I < GetParam(); ++I) {
    ASSERT_TRUE(MaskOne[I]);
  }
}

TEST_P(SubGroupMask, Flip) {
  MaskZero.flip(GetParam() / 2);
  ASSERT_TRUE(MaskZero[GetParam() / 2]);
  MaskOne.flip(0);
  ASSERT_FALSE(MaskOne[0]);
}

TEST_P(SubGroupMask, FindLow) {
  ASSERT_EQ(MaskZero.find_low(), GetParam());
  ASSERT_EQ(MaskOne.find_low(), 0u);
  ASSERT_EQ(MaskAllOnes.find_low(), 0u);
}

TEST_P(SubGroupMask, FindHigh) {
  ASSERT_EQ(MaskZero.find_high(), GetParam());
  ASSERT_EQ(MaskOne.find_high(), 0u);
  ASSERT_EQ(MaskAllOnes.find_high(), GetParam() - 1u);
}

TEST_P(SubGroupMask, ResetLow) {
  MaskAllOnes.reset_low();
  ASSERT_FALSE(MaskAllOnes[0]);

  MaskOne.flip();
  MaskOne.reset_low();
  ASSERT_FALSE(MaskOne[1]);
}

TEST_P(SubGroupMask, ResetHigh) {
  MaskOne.reset_high();
  ASSERT_FALSE(MaskOne[0]);
  MaskAllOnes.reset_high();
  ASSERT_FALSE(MaskAllOnes[GetParam() - 1u]);
}

TEST_P(SubGroupMask, OperatorEq) {
  ASSERT_FALSE(MaskZero == MaskAllOnes);
  MaskZero.flip();
  ASSERT_TRUE(MaskZero == MaskAllOnes);
}

TEST_P(SubGroupMask, OperatorNeq) {
  ASSERT_TRUE(MaskZero != MaskAllOnes);
  MaskZero.flip();
  ASSERT_FALSE(MaskZero != MaskAllOnes);
}

TEST_P(SubGroupMask, OperatorAndEq) {
  MaskZero &= MaskOne;
  ASSERT_TRUE(MaskZero.none());
  MaskAllOnes &= MaskOne;
  ASSERT_EQ(MaskAllOnes, MaskOne);
}

TEST_P(SubGroupMask, OperatorOrEq) {
  MaskZero |= MaskOne;
  ASSERT_EQ(MaskZero, MaskOne);
  MaskAllOnes |= MaskOne;
  ASSERT_TRUE(MaskAllOnes.all());
}

TEST_P(SubGroupMask, OperatorXorEq) {
  MaskZero ^= MaskOne;
  ASSERT_EQ(MaskZero, MaskOne);
  MaskOne ^= MaskAllOnes;
  ASSERT_FALSE(MaskOne[0]);
  for (size_t I = 1; I < GetParam(); ++I) {
    ASSERT_TRUE(MaskOne[I]);
  }
}

TEST_P(SubGroupMask, OperatorShiftLeftEq) {
  size_t Shift = GetParam() / 2;
  MaskAllOnes <<= Shift;
  ASSERT_EQ(MaskAllOnes.find_low(), Shift);
  MaskAllOnes.flip();
  ASSERT_EQ(MaskAllOnes.find_high(), Shift - 1);

  MaskOne <<= Shift;
  ASSERT_EQ(MaskOne.find_high(), MaskOne.find_low());
  ASSERT_TRUE(MaskOne[Shift]);
}

TEST_P(SubGroupMask, OperationShiftRightEq) {
  size_t Shift = GetParam() / 2;
  MaskAllOnes >>= Shift;
  ASSERT_EQ(MaskAllOnes.find_high(), Shift - 1);
  MaskAllOnes.flip();
  ASSERT_EQ(MaskAllOnes.find_low(), Shift);

  MaskOne >>= Shift;
  ASSERT_TRUE(MaskOne.none());
}

TEST_P(SubGroupMask, OperatorShiftLeft) {
  size_t Shift = GetParam() / 2;
  {
    auto Mask = MaskAllOnes << Shift;
    ASSERT_EQ(Mask.find_low(), Shift);
    Mask.flip();
    ASSERT_EQ(Mask.find_high(), Shift - 1);
  }

  {
    auto Mask = MaskOne <<= Shift;
    ASSERT_EQ(Mask.find_high(), Mask.find_low());
    ASSERT_TRUE(Mask[Shift]);
  }
}

TEST_P(SubGroupMask, OperatorShiftRight) {
  size_t Shift = GetParam() / 2;
  {
    auto Mask = MaskAllOnes >> Shift;
    ASSERT_EQ(Mask.find_high(), Shift - 1);
    Mask.flip();
    ASSERT_EQ(Mask.find_low(), Shift);
  }

  {
    auto Mask = MaskOne >> Shift;
    ASSERT_TRUE(Mask.none());
  }
}

TEST_P(SubGroupMask, OperatorBitwiseNot) {
  ASSERT_EQ(~MaskZero, MaskAllOnes);
  ASSERT_EQ(MaskZero, ~MaskAllOnes);
}

TEST_P(SubGroupMask, OperatorAnd) {
  {
    auto Mask = MaskOne & MaskAllOnes;
    ASSERT_EQ(Mask, MaskOne);
  }
  {
    auto Mask = MaskZero & MaskAllOnes;
    ASSERT_EQ(Mask, MaskZero);
  }
}

TEST_P(SubGroupMask, OperatorOr) {
  {
    auto Mask = MaskZero | MaskOne;
    ASSERT_EQ(Mask, MaskOne);
  }
  {
    auto Mask = MaskOne | MaskAllOnes;
    ASSERT_EQ(Mask, MaskAllOnes);
  }
}

TEST_P(SubGroupMask, OperatorXor) {
  {
    auto Mask = MaskZero ^ MaskOne;
    ASSERT_EQ(Mask, MaskOne);
  }
  {
    auto Mask = MaskOne ^ MaskAllOnes;
    ASSERT_FALSE(Mask[0]);
    for (size_t I = 1; I < GetParam(); ++I) {
      ASSERT_TRUE(Mask[I]);
    }
  }
}

TEST_P(SubGroupMask, InsertBitsIntegral) {
  uint8_t Bits = -1;
  size_t Pos = GetParam() / 2u;
  MaskZero.insert_bits(Bits, Pos);
  for (size_t I = 0; I < Pos; ++I) {
    ASSERT_FALSE(MaskZero[I]);
  }
  size_t End = std::min(static_cast<size_t>(MaskZero.size()),
                        Pos + sizeof(Bits) * CHAR_BIT);
  for (size_t I = Pos; I < End; ++I) {
    ASSERT_TRUE(MaskZero[I]);
  }
  for (size_t I = End; I < MaskZero.size(); ++I) {
    ASSERT_FALSE(MaskZero[I]);
  }
}

TEST_P(SubGroupMask, ExtractBitsIntegral) {
  uint8_t Bits = 0;
  size_t Pos = GetParam() / 2u;
  MaskAllOnes.extract_bits(Bits, Pos);
  size_t Mid = std::min(sizeof(Bits) * CHAR_BIT,
                        static_cast<size_t>(MaskAllOnes.size()) - Pos);

  for (size_t I = 0; I < Mid; ++I) {
    ASSERT_TRUE((Bits >> I) & 1u);
  }
  for (size_t I = Mid; I < sizeof(Bits) * CHAR_BIT; ++I) {
    ASSERT_FALSE((Bits >> I) & 1u);
  }
}

TEST_P(SubGroupMask, InsertBitsMarray) {
  sycl::marray<uint8_t, 2> Bits(-1);
  size_t Pos = GetParam() / 2u;
  MaskZero.insert_bits(Bits, Pos);
  for (size_t I = 0; I < Pos; ++I) {
    ASSERT_FALSE(MaskZero[I]);
  }
  size_t End = std::min(static_cast<size_t>(MaskZero.size()),
                        Pos + Bits.size() * CHAR_BIT);
  for (size_t I = Pos; I < End; ++I) {
    ASSERT_TRUE(MaskZero[I]);
  }
  for (size_t I = End; I < MaskZero.size(); ++I) {
    ASSERT_FALSE(MaskZero[I]);
  }
}

TEST_P(SubGroupMask, ExtractBitsMarray) {
  sycl::marray<uint8_t, 2> Bits(0);
  size_t Pos = GetParam() / 2u;
  MaskAllOnes.extract_bits(Bits, Pos);
  size_t Mid = std::min(Bits.size() * CHAR_BIT,
                        static_cast<size_t>(MaskAllOnes.size()) - Pos);

  auto BitsToCheck = *reinterpret_cast<uint16_t *>(Bits.begin());
  for (size_t I = 0; I < Mid; ++I) {
    ASSERT_TRUE((BitsToCheck >> I) & 1u);
  }
  for (size_t I = Mid; I < sizeof(Bits) * CHAR_BIT; ++I) {
    ASSERT_FALSE((BitsToCheck >> I) & 1u);
  }
}

// TODO: sub_group_mask::reference tests
// There is an extension spec update happening in intel/llvm#8174
// TODO: update tests for revision 2 of the extension spec once it is supported

INSTANTIATE_TEST_SUITE_P(OneAPI, SubGroupMask, testing::Values(32, 16, 8));
