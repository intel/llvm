// End-to-end test for clang-offload-wrapper executable:
// Verifies that the clang-offload-wrapper's -batch option correctly processes
// multiple device binaries:
// Test creates two device binary images with associated properties and symbols,
// and a batch input file describing them [1, 1a, 1b, 2].
// It also creates the expected "gold" output for the first image,
// by concatenating the input data [3].
// It then runs clang-offload-wrapper to generate the expected wrapper object,
// and batch file created on step [2] is passed as an input to
// clang-offload-wrapper to produce device binary descriptors for each of the
// wrapped images. Resulting .bc file is compiled with llc to produce an
// object file [4].
// Then the test is compiled and linked with the generated wrapper object [5].
// Finally, the test executable is run and its output is compared to the "gold"
// output created on step [3], ignoring white spaces [6].
// Expected behavior is that the clang-offload-wrapper correctly encodes
// the input data for multiple device binaries described in input batch file
// and that the resulting runtime data
// (device code, properties, and symbols) is accessible and matches the input.
// The test checks both integer and byte array property values, ensuring proper
// decoding and runtime access.

// [1] Prepare test data.
// [1a] Create the first binary image.
// RUN: echo -e -n 'device binary image1\n' > %t.bin
// RUN: echo -e -n '[Category1]\nint_prop1=1|10\n[Category2]\nint_prop2=1|20\n' > %t.props
// RUN: echo -e -n 'kernel1\nkernel2\n' > %t.sym

// [1b] Create the second binary image with byte array property values.
// RUN: echo -e -n 'device binary image2\n' > %t_1.bin
// RUN: echo -e -n '[Category3]\n' > %t_1.props
// RUN: echo -e -n 'kernel1=2|IAAAAAAAAAQA\n' >> %t_1.props
// RUN: echo -e -n 'kernel2=2|oAAAAAAAAAw///3/wB\n' >> %t_1.props

// [2] Create the batch file input for the wrapper.
// RUN: echo '[Code|Properties|Symbols]' > %t.batch
// RUN: echo %t.bin"|"%t.props"|"%t.sym >> %t.batch
// RUN: echo %t_1.bin"|"%t_1.props"|" >> %t.batch

// [3] Generate "gold" output. "gold" output is the concatenation of all input
// data, in the order it is expected to be outputed by the test,
// see `dumpBinary0` below.
// After test is run on step [6], `dumpBinary0` outputs binary image data and
// this output is compared to the "gold" output.
// RUN: cat %t.bin %t.props %t.sym > %t.all

// [4] Create the wrapper object.
// RUN: clang-offload-wrapper -kind=sycl -target=TARGET -format=native -batch %t.batch -o %t.wrapped.bc
// RUN: llc --filetype=obj %t.wrapped.bc -o %t.wrapped.o

// [5] Compile & link the test with the wrapper.
// RUN: %clangxx %t.wrapped.o %s -o %t.batch.exe

// [6] Run and check ignoring white spaces.
// RUN: %t.batch.exe > %t.batch.exe.out
// RUN: diff -b %t.batch.exe.out %t.all

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>

// Data types created by the offload wrapper and inserted in the wrapper object.
// Matches those defined in SYCL Runtime.
struct _sycl_offload_entry_struct {
  void *addr;
  char *name;
  size_t size;
  int32_t flags;
  int32_t reserved;
};

typedef _sycl_offload_entry_struct *_sycl_offload_entry;

struct _sycl_device_binary_property_struct {
  char *Name;       // Null-terminated property name.
  void *ValAddr;    // Address of property value.
  uint32_t Type;    // pi_property_type.
  uint64_t ValSize; // Size of property value in bytes.
};

typedef _sycl_device_binary_property_struct *sycl_device_binary_property;

struct _sycl_device_binary_property_set_struct {
  char *Name;                                // The name.
  sycl_device_binary_property PropertiesBegin; // Array start.
  sycl_device_binary_property PropertiesEnd;   // Array end.
};

typedef _sycl_device_binary_property_set_struct *sycl_device_binary_property_set;

struct sycl_device_binary_struct {
  uint16_t Version;
  uint8_t Kind;   // Type of offload model the binary employs; must be 4 for SYCL.
  uint8_t Format; // Format of the binary data: SPIR-V, LLVM IR bitcode, ...
  const char *DeviceTargetSpec;
  const char *CompileOptions;
  const char *LinkOptions;
  const unsigned char *BinaryStart;
  const unsigned char *BinaryEnd;
  _sycl_offload_entry EntriesBegin;
  _sycl_offload_entry EntriesEnd;
  sycl_device_binary_property_set PropertySetsBegin;
  sycl_device_binary_property_set PropertySetsEnd;
};
typedef sycl_device_binary_struct *sycl_device_binary;

struct sycl_device_binaries_struct {
  uint16_t Version;
  uint16_t NumDeviceBinaries;
  sycl_device_binary DeviceBinaries;
  _sycl_offload_entry *HostEntriesBegin;
  _sycl_offload_entry *HostEntriesEnd;
};
typedef sycl_device_binaries_struct *sycl_device_binaries;

static sycl_device_binaries BinDesc = nullptr;

// Wrapper object has code which calls these 2 functions below.
extern "C" void __sycl_register_lib(sycl_device_binaries desc) {
  BinDesc = desc;
}

extern "C" void __sycl_unregister_lib() {}

#define ASSERT(Cond, Msg)                            \
  if (!(Cond)) {                                     \
    std::cerr << "*** ERROR: wrong " << Msg << "\n"; \
    return 1;                                        \
  }

static std::string getString(const unsigned char *B, const unsigned char *E) {
  return std::string(reinterpret_cast<const char *>(B), E - B);
}

static int getInt(void *Addr) {
  const char *Ptr = reinterpret_cast<const char *>(Addr);
  return Ptr[0] | (Ptr[1] << 8) | (Ptr[2] << 16) | (Ptr[3] << 24);
}

using byte = unsigned char;

static void printProp(const sycl_device_binary_property &Prop) {
  std::cerr << "Property " << Prop->Name << " {\n";
  std::cerr << "  Type: " << Prop->Type << "\n";
  if (Prop->Type != 1)
    std::cerr << "  Size = " << Prop->ValSize << "\n";

  std::cerr << "  Value = ";
  if (Prop->Type == 1)
    std::cerr << getInt(&Prop->ValSize);
  else {
    std::cerr << " {\n   ";

    byte *Ptr = (byte *)Prop->ValAddr;

    for (auto I = 0; I < Prop->ValSize && I < 100; ++I) {
      std::cerr << " 0x" << std::hex << (unsigned int)Ptr[I];
      std::cerr << std::dec;
    }
    std::cerr << "\n  }";
  }
  std::cerr << "\n";
  std::cerr << "}\n";
}

static int dumpBinary0() {
  sycl_device_binary Bin = &BinDesc->DeviceBinaries[0];
  ASSERT(Bin->Kind == 4, "Bin->Kind");
  ASSERT(Bin->Format == 1, "Bin->Format");

  // Dump code.
  std::cout << getString(Bin->BinaryStart, Bin->BinaryEnd);
  // Dump properties.
  for (sycl_device_binary_property_set PropSet = Bin->PropertySetsBegin; PropSet != Bin->PropertySetsEnd; ++PropSet) {
    std::cout << "[" << PropSet->Name << "]"
              << "\n";

    for (sycl_device_binary_property Prop = PropSet->PropertiesBegin; Prop != PropSet->PropertiesEnd; ++Prop) {
      ASSERT(Prop->Type == 1, "Prop->Type");
      std::cout << Prop->Name << "=" << Prop->Type << "|" << getInt(&Prop->ValSize) << "\n";
    }
  }
  // Dump symbols.
  for (_sycl_offload_entry Entry = Bin->EntriesBegin; Entry != Bin->EntriesEnd; ++Entry)
    std::cout << Entry->name << "\n";
  return 0;
}

// Clang offload wrapper does Base64 decoding on byte array property values, so
// they can't be dumped as is and compared to the original. Instead, this
// testcase checks that the byte array in the property value is equal to the
// pre-decoded byte array.
static int checkBinary1() {
  // Decoded from "IAAAAAAAAAQA":
  const byte Arr0[] = {8, 0, 0, 0, 0, 0, 0, 0, 0x1};
  // Decoded from "oAAAAAAAAAw///3/wB":
  const byte Arr1[] = {40, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF, 0x7F, 0xFF, 0x70};

  struct {
    const byte *Ptr;
    const size_t Size;
  } GoldArrays[] = {
      {Arr0, sizeof(Arr0)},
      {Arr1, sizeof(Arr1)}};
  sycl_device_binary Bin = &BinDesc->DeviceBinaries[1];
  ASSERT(Bin->Kind == 4, "Bin->Kind");
  ASSERT(Bin->Format == 1, "Bin->Format");

  for (sycl_device_binary_property_set PropSet = Bin->PropertySetsBegin; PropSet != Bin->PropertySetsEnd; ++PropSet) {
    int Cnt = 0;

    for (sycl_device_binary_property Prop = PropSet->PropertiesBegin; Prop != PropSet->PropertiesEnd; ++Prop, ++Cnt) {
      ASSERT(Prop->Type == 2, "Prop->Type"); // Must be a byte array.
      char *Ptr = reinterpret_cast<char *>(Prop->ValAddr);
      int Cmp = std::memcmp(Prop->ValAddr, GoldArrays[Cnt].Ptr, GoldArrays[Cnt].Size);
      ASSERT(Cmp == 0, "byte array property");
    }
  }
  return 0;
}

int main(int argc, char **argv) {
  ASSERT(BinDesc->NumDeviceBinaries == 2, "BinDesc->NumDeviceBinaries");
  ASSERT(BinDesc->Version == 1, "BinDesc->Version");

  if (dumpBinary0() != 0)
    return 1;
  if (checkBinary1() != 0)
    return 1;
  return 0;
}
