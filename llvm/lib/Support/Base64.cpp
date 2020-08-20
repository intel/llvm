//===--- Base64.cpp - Base64 Encoder/Decoder Implementaion ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Base64.h"

#include <memory>

using namespace llvm;

namespace {

using byte = Base64::byte;

::llvm::Error makeError(const Twine &Msg) {
  return createStringError(std::error_code{}, Msg);
}

class Base64Impl {
private:
  static constexpr char EncodingTable[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                          "abcdefghijklmnopqrstuvwxyz"
                                          "0123456789+/";

  static_assert(sizeof(EncodingTable) == 65, "");

  // Compose an index into the encoder table from two bytes and the number of
  // significant bits in the lower byte until the byte boundary.
  static inline int composeInd(byte ByteLo, byte ByteHi, int BitsLo) {
    int Res = ((ByteHi << BitsLo) | (ByteLo >> (8 - BitsLo))) & 0x3F;
    return Res;
  }

  // Decode a single character.
  static inline int decode(char Ch) {
    if (Ch >= 'A' && Ch <= 'Z') // 0..25
      return Ch - 'A';
    else if (Ch >= 'a' && Ch <= 'z') // 26..51
      return Ch - 'a' + 26;
    else if (Ch >= '0' && Ch <= '9') // 52..61
      return Ch - '0' + 52;
    else if (Ch == '+') // 62
      return 62;
    else if (Ch == '/') // 63
      return 63;
    return -1;
  }

  // Decode a quadruple of characters.
  static inline Expected<bool> decode4(const char *Src, byte *Dst) {
    int BadCh = -1;

    for (auto I = 0; I < 4; ++I) {
      char Ch = Src[I];
      int Byte = decode(Ch);

      if (Byte < 0) {
        BadCh = Ch;
        break;
      }
      Dst[I] = (byte)Byte;
    }
    if (BadCh == -1)
      return true;
    return makeError("invalid char in Base64Impl encoding: 0x" + Twine(BadCh));
  }

public:
  static size_t getEncodedSize(size_t SrcSize) {
    constexpr int ByteSizeInBits = 8;
    constexpr int EncBitsPerChar = 6;
    return (SrcSize * ByteSizeInBits + (EncBitsPerChar - 1)) / EncBitsPerChar;
  }

  static size_t encode(const byte *Src, raw_ostream &Out, size_t SrcSize) {
    size_t Off = 0;

    // encode full byte triples
    for (size_t TriB = 0; TriB < SrcSize / 3; ++TriB) {
      Off = TriB * 3;
      byte Byte0 = Src[Off++];
      byte Byte1 = Src[Off++];
      byte Byte2 = Src[Off++];

      Out << EncodingTable[Byte0 & 0x3F];
      Out << EncodingTable[composeInd(Byte0, Byte1, 2)];
      Out << EncodingTable[composeInd(Byte1, Byte2, 4)];
      Out << EncodingTable[(Byte2 >> 2) & 0x3F];
    }
    // encode the remainder
    int RemBytes = SrcSize - Off;

    if (RemBytes > 0) {
      byte Byte0 = Src[Off + 0];
      Out << EncodingTable[Byte0 & 0x3F];

      if (RemBytes > 1) {
        byte Byte1 = Src[Off + 1];
        Out << EncodingTable[composeInd(Byte0, Byte1, 2)];
        Out << EncodingTable[(Byte1 >> 4) & 0x3F];
      } else {
        Out << EncodingTable[(Byte0 >> 6) & 0x3F];
      }
    }
    return getEncodedSize(SrcSize);
  }

  static size_t getDecodedSize(size_t SrcSize) { return (SrcSize * 3 + 3) / 4; }

  static Expected<size_t> decode(const char *Src, byte *Dst, size_t SrcSize) {
    size_t SrcOff = 0;
    size_t DstOff = 0;

    // decode full quads
    for (size_t Qch = 0; Qch < SrcSize / 4; ++Qch, SrcOff += 4, DstOff += 3) {
      byte Ch[4];
      Expected<bool> TrRes = decode4(Src + SrcOff, Ch);

      if (!TrRes)
        return TrRes.takeError();
      // each quad of chars produces three bytes of output
      Dst[DstOff + 0] = Ch[0] | (Ch[1] << 6);
      Dst[DstOff + 1] = (Ch[1] >> 2) | (Ch[2] << 4);
      Dst[DstOff + 2] = (Ch[2] >> 4) | (Ch[3] << 2);
    }
    auto RemChars = SrcSize - SrcOff;

    if (RemChars == 0)
      return DstOff;
    // decode the remainder; variants:
    // 2 chars remain - produces single byte
    // 3 chars remain - produces two bytes

    if (RemChars != 2 && RemChars != 3)
      return makeError("invalid encoded sequence length");

    int Ch0 = decode(Src[SrcOff++]);
    int Ch1 = decode(Src[SrcOff++]);
    int Ch2 = RemChars == 3 ? decode(Src[SrcOff]) : 0;

    if (Ch0 < 0 || Ch1 < 0 || Ch2 < 0)
      return makeError("invalid characters in the encoded sequence remainder");
    Dst[DstOff++] = Ch0 | (Ch1 << 6);

    if (RemChars == 3)
      Dst[DstOff++] = (Ch1 >> 2) | (Ch2 << 4);
    return DstOff;
  }

  static Expected<std::unique_ptr<byte>> decode(const char *Src,
                                                size_t SrcSize) {
    size_t DstSize = getDecodedSize(SrcSize);
    byte *Dst = new byte[DstSize];
    Expected<size_t> Res = decode(Src, Dst, SrcSize);
    if (!Res)
      return Res.takeError();
    return std::unique_ptr<byte>(Dst);
  }
};

constexpr char Base64Impl::EncodingTable[];

} // anonymous namespace

size_t Base64::getEncodedSize(size_t SrcSize) {
  return Base64Impl::getEncodedSize(SrcSize);
}

size_t Base64::encode(const byte *Src, raw_ostream &Out, size_t SrcSize) {
  return Base64Impl::encode(Src, Out, SrcSize);
}

size_t Base64::getDecodedSize(size_t SrcSize) {
  return Base64Impl::getDecodedSize(SrcSize);
}

Expected<size_t> Base64::decode(const char *Src, byte *Dst, size_t SrcSize) {
  return Base64Impl::decode(Src, Dst, SrcSize);
}

Expected<std::unique_ptr<byte>> Base64::decode(const char *Src,
                                               size_t SrcSize) {
  return Base64Impl::decode(Src, SrcSize);
}
