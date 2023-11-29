//==---------------- bitmap_helpers.h  - DPC++ ESIMD on-device test --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#ifndef ESIMD_EXAMPLES_COMMON_BITMAP_HELPERS_H
#define ESIMD_EXAMPLES_COMMON_BITMAP_HELPERS_H

#ifdef _MSC_VER
#pragma warning(disable : 4996)
#endif // _MSC_VER

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <memory>
#include <string>
#include <sycl/detail/defines.hpp>
#include <vector>

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace intel {
namespace util {
namespace bitmap {

#ifndef PACKED
#ifdef _MSC_VER
#define BEGIN_PACKED __pragma(pack(push, 1))
#define END_PACKED __pragma(pack(pop))
#else
#define BEGIN_PACKED _Pragma("pack(push, 1)")
#define END_PACKED _Pragma("pack(pop)")
#endif
#endif

  BEGIN_PACKED

  struct BitMapFileHeader {
    uint16_t Type;
    uint32_t FileSize;
    uint16_t Res1;
    uint16_t Res2;
    uint32_t PixelOffset;
  };

  struct BitMapInfoHeader {
    uint32_t HeaderSize;
    uint32_t Width;
    uint32_t Height;
    uint16_t Planes;
    uint16_t BitsPerPixel;
    uint32_t Compression;
    uint32_t ImageSize;
    uint32_t XPixelsPerMeter;
    uint32_t YPixelsPerMeter;
    uint32_t ColorsUsed;
    uint32_t ColorsImportant;
  };

  END_PACKED

  class BitMap {
    unsigned Width;
    unsigned Height;
    unsigned BPP;
    unsigned XPPM, YPPM;
    unsigned char *Data;

  protected:
    BitMap(unsigned W, unsigned H, unsigned bpp, unsigned xppm, unsigned yppm)
        : Width(W), Height(H), BPP(bpp), XPPM(xppm), YPPM(yppm), Data(nullptr) {
    }

  public:
    BitMap() : Width(0), Height(0), BPP(0), XPPM(0), YPPM(0), Data(nullptr) {}
    ~BitMap() {
      if (Data) {
        std::free(Data);
        Data = nullptr;
      }
    }

    unsigned getWidth() const { return Width; }
    unsigned getHeight() const { return Height; }
    unsigned getBPP() const { return BPP; }
    unsigned getXPPM() const { return XPPM; }
    unsigned getYPPM() const { return YPPM; }
    unsigned char *getData() const { return Data; }

    void setData(unsigned char *D) {
      if (Data)
        std::free(Data);
      Data = D;
    }

    // Copy except data.
    BitMap(const BitMap &B) {
      Width = B.Width;
      Height = B.Height;
      BPP = B.BPP;
      XPPM = B.XPPM;
      YPPM = B.YPPM;
      Data = nullptr;
    }

    // Move everything.
    BitMap(BitMap &&B) {
      Width = B.Width;
      Height = B.Height;
      BPP = B.BPP;
      XPPM = B.XPPM;
      YPPM = B.YPPM;
      Data = B.Data;
      B.Data = nullptr;
    }

    static BitMap load(const std::string &FilePath) {
      const auto S = FilePath.c_str();
      const auto FP = std::fopen(S, "rb");
      if (!FP) {
        std::perror(S);
        exit(1);
      }

      // Verify bitmap file header.
      BitMapFileHeader BFH;
      if (std::fread(&BFH, 1, sizeof(BFH), FP) != sizeof(BFH)) {
        std::cerr << "Error: failed to read bitmap header from " << S << ".\n";
        exit(1);
      }
      if (BFH.Type != 0x4d42) {
        std::cerr << "Error: " << S << "is not a bitmap file.\n";
        exit(1);
      }

      // Verify bitmap info header.
      BitMapInfoHeader BIH;
      if (std::fread(&BIH, 1, sizeof(BIH), FP) != sizeof(BIH)) {
        std::cerr << "Error: failed to read bitmap info from " << S << ".\n";
        exit(1);
      }
      if (BIH.HeaderSize != 40 || BIH.Planes != 1 || BIH.BitsPerPixel != 24 ||
          BIH.Compression != 0 || BIH.ColorsUsed != 0 ||
          BIH.ColorsImportant != 0) {
        std::cerr << "Error: " << S << " is not supported.\n";
        exit(1);
      }
      if (BIH.ImageSize == 0)
        BIH.ImageSize = BIH.Width * BIH.Height * (BIH.BitsPerPixel / 8);

      unsigned char *Data =
          reinterpret_cast<unsigned char *>(std::malloc(BIH.ImageSize));
      if (!Data) {
        std::cerr << "Error: insufficient memory to load " << S << ".\n";
        exit(1);
      }

      std::fseek(FP, BFH.PixelOffset, SEEK_SET);
      if (std::fread(Data, 1, BIH.ImageSize, FP) != BIH.ImageSize) {
        std::cerr << "Error: failed to read image from " << S << ".\n";
        exit(1);
      }

      std::fclose(FP);

      BitMap Img(BIH.Width, BIH.Height, BIH.BitsPerPixel, BIH.XPixelsPerMeter,
                 BIH.YPixelsPerMeter);
      Img.setData(Data);

      return Img;
    }

    void save(const std::string &FilePath) {
      const auto S = FilePath.c_str();
      const auto FP = std::fopen(S, "wb");
      if (!FP) {
        std::perror(S);
        exit(1);
      }

      unsigned ImageSize = (BPP / 8) * Width * Height;

      BitMapFileHeader BFH;
      BitMapInfoHeader BIH;

      BFH.Type = 0x4d42;
      BFH.FileSize = sizeof(BFH) + sizeof(BIH) + ImageSize;
      BFH.Res1 = BFH.Res2 = 0;
      BFH.PixelOffset = sizeof(BFH) + sizeof(BIH);

      BIH.HeaderSize = sizeof(BIH);
      BIH.Width = Width;
      BIH.Height = Height;
      BIH.Planes = 1;
      BIH.BitsPerPixel = BPP;
      BIH.Compression = 0;
      BIH.ImageSize = ImageSize;
      BIH.XPixelsPerMeter = XPPM;
      BIH.YPixelsPerMeter = YPPM;
      BIH.ColorsUsed = 0;
      BIH.ColorsImportant = 0;

      std::fwrite(&BFH, sizeof(BFH), 1, FP);
      std::fwrite(&BIH, sizeof(BIH), 1, FP);
      std::fwrite(Data, 1, ImageSize, FP);

      std::fclose(FP);
    }

    template <typename T>
    static bool checkResult(const char *f_out_str, const char *f_gold_str,
                            T tolerance) {
      unsigned char header_out[54];
      unsigned char header_gold[54];
      unsigned int width;
      unsigned int height;
      unsigned char *img_out, *img_gold;
      unsigned int i;
      FILE *f_out;
      FILE *f_gold;

      f_out = fopen(f_out_str, "rb");
      if (f_out == NULL) {
        perror(f_out_str);
        return false;
      }
      if (fread(header_out, 1, 54, f_out) != 54) {
        perror(f_out_str);
        return false;
      }

      f_gold = fopen(f_gold_str, "rb");
      if (f_gold == NULL) {
        perror(f_gold_str);
        return false;
      }
      if (fread(header_gold, 1, 54, f_gold) != 54) {
        perror(f_gold_str);
        return false;
      }

      if (header_out[18] != header_gold[18] ||
          header_out[22] != header_gold[22]) {
        fclose(f_out);
        fclose(f_gold);
        perror("headers are different\n");
        return false;
      }

      width = std::abs(*(short *)&header_out[18]);
      height = std::abs(*(short *)&header_out[22]);

      auto img_out_vector = std::vector<unsigned char>(width * height * 3);
      auto img_gold_vector = std::vector<unsigned char>(width * height * 3);
      img_out = img_out_vector.data();
      img_gold = img_gold_vector.data();

      if (fread(img_out, 1, width * height * 3, f_out) != width * height * 3) {
        perror(f_out_str);
        return false;
      }

      if (fread(img_gold, 1, width * height * 3, f_gold) !=
          width * height * 3) {
        perror(f_gold_str);
        return false;
      }

      fclose(f_out);
      fclose(f_gold);

      for (i = 0; i < width * height * 3; i++) {
        if (std::abs(img_out[i] - img_gold[i]) > tolerance) {
          return false;
        }
      }

      return true;
    }
  };

  } // end namespace bitmap
  } // end namespace util
  } // end namespace intel
  } // end namespace ext
  } // namespace _V1
} // namespace sycl

#endif // ESIMD_EXAMPLES_COMMON_BITMAP_HELPERS_H
