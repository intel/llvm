//==--- MDParsing.h - Helper to retrieve fusion information from metadata --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_FUSION_PASSES_MDPARSING_H
#define SYCL_FUSION_PASSES_MDPARSING_H

namespace llvm {
template <typename T> inline Expected<T> metadataToUInt(const Metadata *MD) {
  if (auto *C = dyn_cast<ConstantAsMetadata>(MD)) {
    return static_cast<T>(C->getValue()->getUniqueInteger().getZExtValue());
  }
  {
    std::string Msg{"Invalid metadata format: "};
    raw_string_ostream MsgSS{Msg};
    return createStringError(inconvertibleErrorCode(), MsgSS.str());
  }
}

inline static Expected<SmallVector<unsigned char>>
decodeConstantMetadata(const Metadata *MD) {
  const auto *MDS = dyn_cast<MDString>(MD);
  if (!MDS) {
    return createStringError(inconvertibleErrorCode(),
                             "Invalid metadata format - not constant");
  }
  SmallVector<unsigned char> Values;
  Values.resize_for_overwrite(MDS->getLength());
  std::memcpy(Values.data(), MDS->getString().data(), MDS->getLength());
  return Values;
}

} // namespace llvm

#endif // SYCL_FUSION_PASSES_MDPARSING_H
