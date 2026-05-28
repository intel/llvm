//===--- Base64.h - Base64 Encoder/Decoder ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Adjusted copy of llvm/include/llvm/Support/Base64.h.
// TODO: Remove once we can consistently link the SYCL runtime library with
// LLVMSupport.

#pragma once

#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace sycl {
inline namespace _V1 {
namespace detail {

class Base64 {
private:
  // Encoding table: 6-bit index -> base64 character.
  //
  // Base64 packs every 3 bytes (24 bits) of input into 4 output characters,
  // each carrying 6 bits of the input. The table indexes by the 6-bit value
  // and produces the corresponding character per RFC 4648:
  //
  //   index  0..25  -> 'A'..'Z'
  //   index 26..51  -> 'a'..'z'
  //   index 52..61  -> '0'..'9'
  //   index 62      -> '+'
  //   index 63      -> '/'
  //
  // Used by encode() only. decode() uses arithmetic on the input character
  // (see below) since its 7-bit ASCII input domain is sparse and a reverse
  // lookup table would be mostly invalid entries.
  static constexpr char EncodingTable[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                          "abcdefghijklmnopqrstuvwxyz"
                                          "0123456789+/";

  // Compose a 6-bit base64 index that straddles a byte boundary in the input
  // stream. \p ByteLo holds the most-significant bits of the index in its
  // \p BitsLo low bits; \p ByteHi holds the least-significant bits of the
  // index in its high (8 - \p BitsLo) bits. The two are stitched together and
  // masked to 6 bits to yield a value in [0, 63] suitable for indexing into
  // EncodingTable. Used by encode() to handle the 2nd and 3rd output
  // characters of every 3-byte input triple, where the 6-bit slice is not
  // byte-aligned.
  static inline int composeInd(uint8_t ByteLo, uint8_t ByteHi, int BitsLo) {
    return ((ByteHi << BitsLo) | (ByteLo >> (8 - BitsLo))) & 0x3F;
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
  static inline void decode4(const char *Src, byte *Dst) {
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
    if (BadCh != -1)
      throw sycl::exception(make_error_code(errc::invalid),
                            "Invalid char in base 64 encoding.");
  }

public:
  using byte = uint8_t;

  // Get the size of the encoded byte sequence of given size.
  static size_t getEncodedSize(size_t SrcSize) {
    constexpr int ByteSizeInBits = 8;
    constexpr int EncBitsPerChar = 6;
    return (SrcSize * ByteSizeInBits + (EncBitsPerChar - 1)) / EncBitsPerChar;
  }

  // Encode a byte sequence of given size into the output stream.
  // Returns the number of bytes in the encoded result.
  static size_t encode(const byte *Src, std::ostream &Out, size_t SrcSize) {
    size_t Off = 0;

    // Encode full byte triples.
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
    // Encode the remainder, if any.
    int RemBytes = static_cast<int>(SrcSize - Off);

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

  // Get the size of the encoded byte sequence of given size.
  static size_t getDecodedSize(size_t SrcSize) { return (SrcSize * 3 + 3) / 4; }

  // Decode a sequence of given size into a pre-allocated memory.
  // Returns the number of bytes in the decoded result or 0 in case of error.
  static size_t decode(const char *Src, byte *Dst, size_t SrcSize) {
    size_t SrcOff = 0;
    size_t DstOff = 0;

    // decode full quads
    for (size_t Qch = 0; Qch < SrcSize / 4; ++Qch, SrcOff += 4, DstOff += 3) {
      byte Ch[4] = {0, 0, 0, 0};
      decode4(Src + SrcOff, Ch);

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
      throw sycl::exception(make_error_code(errc::invalid),
                            "Invalid encoded sequence length.");

    int Ch0 = decode(Src[SrcOff++]);
    int Ch1 = decode(Src[SrcOff++]);
    int Ch2 = RemChars == 3 ? decode(Src[SrcOff]) : 0;

    if (Ch0 < 0 || Ch1 < 0 || Ch2 < 0)
      throw sycl::exception(
          make_error_code(errc::invalid),
          "Invalid characters in the encoded sequence remainder.");
    Dst[DstOff++] = Ch0 | (Ch1 << 6);

    if (RemChars == 3)
      Dst[DstOff++] = (Ch1 >> 2) | (Ch2 << 4);
    return DstOff;
  }

  // Allocate minimum required amount of memory and decode a sequence of given
  // size into it.
  // Returns the decoded result. The size can be obtained via getDecodedSize.
  static std::unique_ptr<byte[]> decode(const char *Src, size_t SrcSize) {
    size_t DstSize = getDecodedSize(SrcSize);
    std::unique_ptr<byte[]> Dst(new byte[DstSize]);
    decode(Src, Dst.get(), SrcSize);
    return Dst;
  }
};

} // namespace detail
} // namespace _V1
} // namespace sycl
