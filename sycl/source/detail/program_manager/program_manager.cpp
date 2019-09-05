//==------ program_manager.cpp --- SYCL program manager---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/detail/program_manager/program_manager.hpp>
#include <CL/sycl/detail/type_traits.hpp>
#include <CL/sycl/detail/util.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/stl.hpp>

#include <cassert>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>

namespace cl {
namespace sycl {
namespace detail {

static constexpr int DbgProgMgr = 0;

ProgramManager &ProgramManager::getInstance() {
  // The singleton ProgramManager instance, uses the "magic static" idiom.
  static ProgramManager Instance;
  return Instance;
}

static RT::PiDevice getFirstDevice(RT::PiContext Context) {
  cl_uint NumDevices = 0;
  PI_CALL(RT::piContextGetInfo(Context, PI_CONTEXT_INFO_NUM_DEVICES,
                               sizeof(NumDevices), &NumDevices,
                               /*param_value_size_ret=*/nullptr));
  assert(NumDevices > 0 && "Context without devices?");

  vector_class<RT::PiDevice> Devices(NumDevices);
  size_t ParamValueSize = 0;
  PI_CALL(RT::piContextGetInfo(Context, PI_CONTEXT_INFO_DEVICES,
                               sizeof(cl_device_id) * NumDevices, &Devices[0],
                               &ParamValueSize));
  assert(ParamValueSize == sizeof(cl_device_id) * NumDevices &&
         "Number of CL_CONTEXT_DEVICES should match CL_CONTEXT_NUM_DEVICES.");
  return Devices[0];
}

static RT::PiProgram createBinaryProgram(const RT::PiContext Context,
                                         const unsigned char *Data,
                                         size_t DataLen) {
  // FIXME: we don't yet support multiple devices with a single binary.
#ifndef _NDEBUG
  cl_uint NumDevices = 0;
  PI_CALL(RT::piContextGetInfo(Context, PI_CONTEXT_INFO_NUM_DEVICES,
                                  sizeof(NumDevices), &NumDevices,
                                  /*param_value_size_ret=*/nullptr));
  assert(NumDevices > 0 &&
         "Only a single device is supported for AOT compilation");
#endif

  RT::PiDevice Device = getFirstDevice(Context);
  RT::PiResult Err = PI_SUCCESS;
  pi_int32 BinaryStatus = CL_SUCCESS;
  RT::PiProgram Program;
  PI_CALL((Program = RT::piclProgramCreateWithBinary(
      Context, 1 /*one binary*/, &Device,
      &DataLen, &Data, &BinaryStatus, &Err), Err));
  return Program;
}

static RT::PiProgram createSpirvProgram(const RT::PiContext Context,
                                        const unsigned char *Data,
                                        size_t DataLen) {
  RT::PiProgram Program = nullptr;
  PI_CALL(pi::piProgramCreate(Context, Data, DataLen, &Program));
  return Program;
}

RT::PiProgram ProgramManager::getBuiltOpenCLProgram(OSModuleHandle M,
                                                    const context &Context) {
  std::shared_ptr<context_impl> Ctx = getSyclObjImpl(Context);
  std::map<OSModuleHandle, RT::PiProgram> &CachedPrograms =
      Ctx->getCachedPrograms();
  auto It = CachedPrograms.find(M);
  if (It != CachedPrograms.end())
    return It->second;

  DeviceImage *Img = nullptr;
  using PiProgramT = remove_pointer_t<RT::PiProgram>;
  unique_ptr_class<PiProgramT, decltype(RT::piProgramRelease)> ProgramManaged(
      loadProgram(M, Context, &Img), RT::piProgramRelease);

  build(ProgramManaged.get(), Img->BuildOptions);
  RT::PiProgram Program = ProgramManaged.release();
  CachedPrograms[M] = Program;

  return Program;
}

RT::PiKernel ProgramManager::getOrCreateKernel(OSModuleHandle M,
                                               const context &Context,
                                               const string_class &KernelName) {
  if (DbgProgMgr > 0) {
    std::cerr << ">>> ProgramManager::getOrCreateKernel(" << M << ", "
              << getRawSyclObjImpl(Context) << ", " << KernelName << ")\n";
  }
  RT::PiProgram Program = getBuiltOpenCLProgram(M, Context);
  std::shared_ptr<context_impl> Ctx = getSyclObjImpl(Context);
  std::map<RT::PiProgram, std::map<string_class, RT::PiKernel>> &CachedKernels =
      Ctx->getCachedKernels();
  std::map<string_class, RT::PiKernel> &KernelsCache = CachedKernels[Program];
  RT::PiKernel &Kernel = KernelsCache[KernelName];
  if (!Kernel) {
    RT::PiResult Err = PI_SUCCESS;
    PI_CALL((Kernel = RT::piKernelCreate(
        Program, KernelName.c_str(), &Err), Err));
  }
  return Kernel;
}

RT::PiProgram ProgramManager::getClProgramFromClKernel(RT::PiKernel Kernel) {
  RT::PiProgram Program;
  PI_CALL(RT::piKernelGetInfo(
      Kernel, CL_KERNEL_PROGRAM, sizeof(cl_program), &Program, nullptr));
  return Program;
}

string_class ProgramManager::getProgramBuildLog(const RT::PiProgram &Program) {
  size_t Size = 0;
  PI_CALL(RT::piProgramGetInfo(Program, CL_PROGRAM_DEVICES, 0, nullptr, &Size));
  vector_class<RT::PiDevice> PIDevices(Size / sizeof(RT::PiDevice));
  PI_CALL(RT::piProgramGetInfo(Program, CL_PROGRAM_DEVICES, Size,
                               PIDevices.data(), nullptr));
  string_class Log = "The program was built for " +
                     std::to_string(PIDevices.size()) + " devices";
  for (RT::PiDevice &Device : PIDevices) {
    PI_CALL(RT::piProgramGetBuildInfo(Program, Device, CL_PROGRAM_BUILD_LOG, 0,
                                      nullptr, &Size));
    vector_class<char> DeviceBuildInfo(Size);
    PI_CALL(RT::piProgramGetBuildInfo(Program, Device, CL_PROGRAM_BUILD_LOG,
                                      Size, DeviceBuildInfo.data(), nullptr));
    PI_CALL(
        RT::piDeviceGetInfo(Device, PI_DEVICE_INFO_NAME, 0, nullptr, &Size));
    vector_class<char> DeviceName(Size);
    PI_CALL(RT::piDeviceGetInfo(Device, PI_DEVICE_INFO_NAME, Size,
                                DeviceName.data(), nullptr));

    Log += "\nBuild program log for '" + string_class(DeviceName.data()) +
           "':\n" + string_class(DeviceBuildInfo.data());
  }
  return Log;
}

void ProgramManager::build(RT::PiProgram Program, const string_class &Options,
                           std::vector<RT::PiDevice> Devices) {

  if (DbgProgMgr > 0) {
    std::cerr << ">>> ProgramManager::build(" << Program << ", " << Options
              << ", ... " << Devices.size() << ")\n";
  }
  const char *Opts = std::getenv("SYCL_PROGRAM_BUILD_OPTIONS");

  for (const auto &DeviceId : Devices) {
    if (!createSyclObjFromImpl<device>(std::make_shared<device_impl_pi>(DeviceId)).
            get_info<info::device::is_compiler_available>()) {
      throw feature_not_supported(
          "Online compilation is not supported by this device");
    }
  }

  if (!Opts)
    Opts = Options.c_str();
  if (PI_CALL_RESULT(RT::piProgramBuild(
        Program, Devices.size(), Devices.data(),
        Opts, nullptr, nullptr)) == PI_SUCCESS)
    return;

  throw compile_program_error(getProgramBuildLog(Program));
}

void ProgramManager::addImages(pi_device_binaries DeviceBinary) {
  std::lock_guard<std::mutex> Guard(Sync::getGlobalLock());

  for (int I = 0; I < DeviceBinary->NumDeviceBinaries; I++) {
    pi_device_binary Img = &(DeviceBinary->DeviceBinaries[I]);
    OSModuleHandle M = OSUtil::getOSModuleHandle(Img);
    auto &Imgs = m_DeviceImages[M];

    if (Imgs == nullptr)
      Imgs.reset(new std::vector<DeviceImage *>());
    Imgs->push_back(Img);
  }
}

void ProgramManager::debugDumpBinaryImage(const DeviceImage *Img) const {
  std::cerr << "  --- Image " << Img << "\n";
  if (!Img)
    return;
  std::cerr << "    Version  : " << (int)Img->Version << "\n";
  std::cerr << "    Kind     : " << (int)Img->Kind << "\n";
  std::cerr << "    Format   : " << (int)Img->Format << "\n";
  std::cerr << "    Target   : " << Img->DeviceTargetSpec << "\n";
  std::cerr << "    Options  : "
            << (Img->BuildOptions ? Img->BuildOptions : "NULL") << "\n";
  std::cerr << "    Bin size : "
            << ((intptr_t)Img->BinaryEnd - (intptr_t)Img->BinaryStart) << "\n";
}

void ProgramManager::debugDumpBinaryImages() const {
  for (const auto &ModImgvec : m_DeviceImages) {
    std::cerr << "  ++++++ Module: " << ModImgvec.first << "\n";
    for (const auto *Img : *(ModImgvec.second)) {
      debugDumpBinaryImage(Img);
    }
  }
}

struct ImageDeleter {
  void operator()(DeviceImage *I) {
    delete[] I->BinaryStart;
    delete I;
  }
};

static bool is_device_binary_type_supported(const context &C,
                                  RT::PiDeviceBinaryType Format) {
  // All formats except PI_DEVICE_BINARY_TYPE_SPIRV are supported.
  if (Format != PI_DEVICE_BINARY_TYPE_SPIRV)
    return true;

  // OpenCL 2.1 and greater require clCreateProgramWithIL
  if (pi::useBackend(pi::SYCL_BE_PI_OPENCL) &&
      C.get_platform().get_info<info::platform::version>() >= "2.1")
    return true;

  // Otherwise we need cl_khr_il_program extension to be present
  // and we can call clCreateProgramWithILKHR using the extension
  for (const auto &D : C.get_devices()) {
    auto Extensions = D.get_info<info::device::extensions>();
    if (std::find(Extensions.begin(), Extensions.end(),
                  string_class("cl_khr_il_program")) != Extensions.end())
      return true;
  }

  // This device binary type is not supported.
  return false;
}

RT::PiProgram ProgramManager::loadProgram(OSModuleHandle M,
                                          const context &Context,
                                          DeviceImage **I) {
  std::lock_guard<std::mutex> Guard(Sync::getGlobalLock());

  if (DbgProgMgr > 0) {
    std::cerr << ">>> ProgramManager::loadProgram(" << M << ","
              << getRawSyclObjImpl(Context) << ")\n";
  }

  DeviceImage *Img = nullptr;
  bool UseKernelSpv = false;
  const std::string UseSpvEnv("SYCL_USE_KERNEL_SPV");

  if (const char *Spv = std::getenv(UseSpvEnv.c_str())) {
    // The env var requests that the program is loaded from a SPIRV file on disk
    UseKernelSpv = true;
    std::string Fname(Spv);
    std::ifstream File(Fname, std::ios::binary);

    if (!File.is_open()) {
      throw runtime_error(std::string("Can't open file specified via ") +
                          UseSpvEnv + ": " + Fname);
    }
    File.seekg(0, std::ios::end);
    size_t Size = File.tellg();
    auto *Data = new unsigned char[Size];
    File.seekg(0);
    File.read(reinterpret_cast<char *>(Data), Size);
    File.close();

    if (!File.good()) {
      delete[] Data;
      throw runtime_error(std::string("read from ") + Fname +
                          std::string(" failed"));
    }
    Img = new DeviceImage();
    Img->Version          = PI_DEVICE_BINARY_VERSION;
    Img->Kind             = PI_DEVICE_BINARY_OFFLOAD_KIND_SYCL;
    Img->Format           = PI_DEVICE_BINARY_TYPE_NONE;
    Img->DeviceTargetSpec = PI_DEVICE_BINARY_TARGET_UNKNOWN;
    Img->BuildOptions = "";
    Img->ManifestStart = nullptr;
    Img->ManifestEnd = nullptr;
    Img->BinaryStart = Data;
    Img->BinaryEnd = Data + Size;
    Img->EntriesBegin = nullptr;
    Img->EntriesEnd = nullptr;

    std::unique_ptr<DeviceImage, ImageDeleter> ImgPtr(Img, ImageDeleter());
    m_OrphanDeviceImages.emplace_back(std::move(ImgPtr));

    if (DbgProgMgr > 0) {
      std::cerr << "loaded device image from " << Fname << "\n";
    }
  } else {
    // Take all device images in module M and ask the native runtime under the
    // given context to choose one it prefers.
    auto ImgIt = m_DeviceImages.find(M);

    if (ImgIt == m_DeviceImages.end()) {
      throw runtime_error("No device program image found");
    }
    std::vector<DeviceImage *> *Imgs = (ImgIt->second).get();

    PI_CALL(RT::piextDeviceSelectBinary(
      0, Imgs->data(), (cl_uint)Imgs->size(), &Img));

    if (DbgProgMgr > 0) {
      std::cerr << "available device images:\n";
      debugDumpBinaryImages();
      std::cerr << "selected device image: " << Img << "\n";
      debugDumpBinaryImage(Img);
    }
  }
  // perform minimal sanity checks on the device image and the descriptor
  if (Img->BinaryEnd < Img->BinaryStart) {
    throw runtime_error("Malformed device program image descriptor");
  }
  if (Img->BinaryEnd == Img->BinaryStart) {
    throw runtime_error("Invalid device program image: size is zero");
  }
  size_t ImgSize = static_cast<size_t>(Img->BinaryEnd - Img->BinaryStart);
  auto Format = pi::cast<RT::PiDeviceBinaryType>(Img->Format);

  // Determine the format of the image if not set already
  if (Format == PI_DEVICE_BINARY_TYPE_NONE) {
    struct {
      RT::PiDeviceBinaryType Fmt;
      const uint32_t Magic;
    } Fmts[] = {{PI_DEVICE_BINARY_TYPE_SPIRV, 0x07230203},
                {PI_DEVICE_BINARY_TYPE_LLVMIR_BITCODE, 0xDEC04342}};
    if (ImgSize >= sizeof(Fmts[0].Magic)) {
      std::remove_const<decltype(Fmts[0].Magic)>::type Hdr = 0;
      std::copy(Img->BinaryStart, Img->BinaryStart + sizeof(Hdr),
                reinterpret_cast<char *>(&Hdr));

      for (const auto &Fmt : Fmts) {
        if (Hdr == Fmt.Magic) {
          Format = Fmt.Fmt;

          // Image binary format wasn't set but determined above - update it;
          if (UseKernelSpv) {
            Img->Format = Format;
          } else {
            // TODO the binary image is a part of the fat binary, the clang
            //   driver should have set proper format option to the
            //   clang-offload-wrapper. The fix depends on AOT compilation
            //   implementation, so will be implemented together with it.
            //   Img->Format can't be updated as it is inside of the in-memory
            //   OS module binary.
            // throw runtime_error("Image format not set");
          }
          if (DbgProgMgr > 1) {
            std::cerr << "determined image format: " << (int)Format << "\n";
          }
          break;
        }
      }
    }
  }
  // Dump program image if requested
  if (std::getenv("SYCL_DUMP_IMAGES") && !UseKernelSpv) {
    std::string Fname("sycl_");
    Fname += Img->DeviceTargetSpec;
    std::string Ext;

    if (Format == PI_DEVICE_BINARY_TYPE_SPIRV) {
      Ext = ".spv";
    } else if (Format == PI_DEVICE_BINARY_TYPE_LLVMIR_BITCODE) {
      Ext = ".bc";
    } else {
      Ext = ".bin";
    }
    Fname += Ext;

    std::ofstream F(Fname, std::ios::binary);

    if (!F.is_open()) {
      throw runtime_error(std::string("Can not write ") + Fname);
    }
    F.write(reinterpret_cast<const char *>(Img->BinaryStart), ImgSize);
    F.close();
  }
  // Load the selected image
  if (!is_device_binary_type_supported(Context, Format))
    throw feature_not_supported("Online compilation is not supported in this context");
  const RT::PiContext &Ctx = getRawSyclObjImpl(Context)->getHandleRef();
  RT::PiProgram Res = nullptr;
  Res = Format == PI_DEVICE_BINARY_TYPE_SPIRV
            ? createSpirvProgram(Ctx, Img->BinaryStart, ImgSize)
            : createBinaryProgram(Ctx, Img->BinaryStart, ImgSize);

  if (I)
    *I = Img;
  if (DbgProgMgr > 1) {
    std::cerr << "created native program: " << Res << "\n";
  }
  return Res;
}

} // namespace detail
} // namespace sycl
} // namespace cl

extern "C" void __tgt_register_lib(pi_device_binaries desc) {
  cl::sycl::detail::ProgramManager::getInstance().addImages(desc);
}

// Executed as a part of current module's (.exe, .dll) static initialization
extern "C" void __tgt_unregister_lib(pi_device_binaries desc) {
  // TODO implement the function
}
