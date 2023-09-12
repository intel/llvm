#include "llvm/IR/FPAccuracy.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(AccuracyLookupTest, TestScalarFloatAccuracy) {
  LLVMContext C;
  Type *FloatTy = Type::getFloatTy(C);

  StringRef Accuracy;

  Accuracy = fp::getAccuracyForFPBuiltin(Intrinsic::fpbuiltin_cos, FloatTy,
                                         fp::FPAccuracy::High);
  EXPECT_EQ("1.0", Accuracy);

  Accuracy = fp::getAccuracyForFPBuiltin(Intrinsic::fpbuiltin_cos, FloatTy,
                                         fp::FPAccuracy::Medium);
  EXPECT_EQ("4.0", Accuracy);

  Accuracy = fp::getAccuracyForFPBuiltin(Intrinsic::fpbuiltin_cos, FloatTy,
                                         fp::FPAccuracy::Low);
  EXPECT_EQ("8192.0", Accuracy);

  Accuracy = fp::getAccuracyForFPBuiltin(Intrinsic::fpbuiltin_tan, FloatTy,
                                         fp::FPAccuracy::SYCL);
  EXPECT_EQ("5.0", Accuracy);

  Accuracy = fp::getAccuracyForFPBuiltin(Intrinsic::fpbuiltin_tan, FloatTy,
                                         fp::FPAccuracy::CUDA);
  EXPECT_EQ("4.0", Accuracy);
}

TEST(AccuracyLookupTest, TestScalarDoubleAccuracy) {
  LLVMContext C;
  Type *DoubleTy = Type::getDoubleTy(C);

  StringRef Accuracy;

  Accuracy = fp::getAccuracyForFPBuiltin(Intrinsic::fpbuiltin_cos, DoubleTy,
                                         fp::FPAccuracy::High);
  EXPECT_EQ("1.0", Accuracy);

  Accuracy = fp::getAccuracyForFPBuiltin(Intrinsic::fpbuiltin_cos, DoubleTy,
                                         fp::FPAccuracy::Medium);
  EXPECT_EQ("4.0", Accuracy);

  Accuracy = fp::getAccuracyForFPBuiltin(Intrinsic::fpbuiltin_cos, DoubleTy,
                                         fp::FPAccuracy::Low);
  // 67108864.0f == 2^(53-26-1) == 26-bits of accuracy
  EXPECT_EQ("67108864.0", Accuracy);

  Accuracy = fp::getAccuracyForFPBuiltin(Intrinsic::fpbuiltin_tan, DoubleTy,
                                         fp::FPAccuracy::SYCL);
  EXPECT_EQ("5.0", Accuracy);

  Accuracy = fp::getAccuracyForFPBuiltin(Intrinsic::fpbuiltin_tan, DoubleTy,
                                         fp::FPAccuracy::CUDA);
  EXPECT_EQ("2.0", Accuracy);
}

TEST(AccuracyLookupTest, TestFixedVectorFloatAccuracy) {
  LLVMContext C;
  Type *VecFloatTy = VectorType::get(Type::getFloatTy(C), 4, false);

  StringRef Accuracy;

  Accuracy = fp::getAccuracyForFPBuiltin(Intrinsic::fpbuiltin_cos, VecFloatTy,
                                         fp::FPAccuracy::High);
  EXPECT_EQ("1.0", Accuracy);

  Accuracy = fp::getAccuracyForFPBuiltin(Intrinsic::fpbuiltin_cos, VecFloatTy,
                                         fp::FPAccuracy::Medium);
  EXPECT_EQ("4.0", Accuracy);

  Accuracy = fp::getAccuracyForFPBuiltin(Intrinsic::fpbuiltin_cos, VecFloatTy,
                                         fp::FPAccuracy::Low);
  EXPECT_EQ("8192.0", Accuracy);

  Accuracy = fp::getAccuracyForFPBuiltin(Intrinsic::fpbuiltin_tan, VecFloatTy,
                                         fp::FPAccuracy::SYCL);
  EXPECT_EQ("5.0", Accuracy);

  Accuracy = fp::getAccuracyForFPBuiltin(Intrinsic::fpbuiltin_tan, VecFloatTy,
                                         fp::FPAccuracy::CUDA);
  EXPECT_EQ("4.0", Accuracy);
}

TEST(AccuracyLookupTest, TestFixedVectorDoubleAccuracy) {
  LLVMContext C;
  Type *VecDoubleTy = VectorType::get(Type::getDoubleTy(C), 4, false);

  StringRef Accuracy;

  Accuracy = fp::getAccuracyForFPBuiltin(Intrinsic::fpbuiltin_cos, VecDoubleTy,
                                         fp::FPAccuracy::High);
  EXPECT_EQ("1.0", Accuracy);

  Accuracy = fp::getAccuracyForFPBuiltin(Intrinsic::fpbuiltin_cos, VecDoubleTy,
                                         fp::FPAccuracy::Medium);
  EXPECT_EQ("4.0", Accuracy);

  Accuracy = fp::getAccuracyForFPBuiltin(Intrinsic::fpbuiltin_cos, VecDoubleTy,
                                         fp::FPAccuracy::Low);
  // 67108864.0f == 2^(53-26-1) == 26-bits of accuracy
  EXPECT_EQ("67108864.0", Accuracy);

  Accuracy = fp::getAccuracyForFPBuiltin(Intrinsic::fpbuiltin_tan, VecDoubleTy,
                                         fp::FPAccuracy::SYCL);
  EXPECT_EQ("5.0", Accuracy);

  Accuracy = fp::getAccuracyForFPBuiltin(Intrinsic::fpbuiltin_tan, VecDoubleTy,
                                         fp::FPAccuracy::CUDA);
  EXPECT_EQ("2.0", Accuracy);
}

TEST(AccuracyLookupTest, TestScalableVectorFloatAccuracy) {
  LLVMContext C;
  Type *VecFloatTy = VectorType::get(Type::getFloatTy(C), 4, true);

  StringRef Accuracy;

  Accuracy = fp::getAccuracyForFPBuiltin(Intrinsic::fpbuiltin_cos, VecFloatTy,
                                         fp::FPAccuracy::High);
  EXPECT_EQ("1.0", Accuracy);

  Accuracy = fp::getAccuracyForFPBuiltin(Intrinsic::fpbuiltin_cos, VecFloatTy,
                                         fp::FPAccuracy::Medium);
  EXPECT_EQ("4.0", Accuracy);

  Accuracy = fp::getAccuracyForFPBuiltin(Intrinsic::fpbuiltin_cos, VecFloatTy,
                                         fp::FPAccuracy::Low);
  EXPECT_EQ("8192.0", Accuracy);

  Accuracy = fp::getAccuracyForFPBuiltin(Intrinsic::fpbuiltin_tan, VecFloatTy,
                                         fp::FPAccuracy::SYCL);
  EXPECT_EQ("5.0", Accuracy);

  Accuracy = fp::getAccuracyForFPBuiltin(Intrinsic::fpbuiltin_tan, VecFloatTy,
                                         fp::FPAccuracy::CUDA);
  EXPECT_EQ("4.0", Accuracy);
}

TEST(AccuracyLookupTest, TestScalableVectorDoubleAccuracy) {
  LLVMContext C;
  Type *VecDoubleTy = VectorType::get(Type::getDoubleTy(C), 4, true);

  StringRef Accuracy;

  Accuracy = fp::getAccuracyForFPBuiltin(Intrinsic::fpbuiltin_cos, VecDoubleTy,
                                         fp::FPAccuracy::High);
  EXPECT_EQ("1.0", Accuracy);

  Accuracy = fp::getAccuracyForFPBuiltin(Intrinsic::fpbuiltin_cos, VecDoubleTy,
                                         fp::FPAccuracy::Medium);
  EXPECT_EQ("4.0", Accuracy);

  Accuracy = fp::getAccuracyForFPBuiltin(Intrinsic::fpbuiltin_cos, VecDoubleTy,
                                         fp::FPAccuracy::Low);
  // 67108864.0f == 2^(53-26-1) == 26-bits of accuracy
  EXPECT_EQ("67108864.0", Accuracy);

  Accuracy = fp::getAccuracyForFPBuiltin(Intrinsic::fpbuiltin_tan, VecDoubleTy,
                                         fp::FPAccuracy::SYCL);
  EXPECT_EQ("5.0", Accuracy);

  Accuracy = fp::getAccuracyForFPBuiltin(Intrinsic::fpbuiltin_tan, VecDoubleTy,
                                         fp::FPAccuracy::CUDA);
  EXPECT_EQ("2.0", Accuracy);
}

} // end namespace
