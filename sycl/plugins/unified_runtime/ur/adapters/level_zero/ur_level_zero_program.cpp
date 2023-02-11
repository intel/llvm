//===--------- ur_level_zero_program.cpp - Level Zero Adapter ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "ur_level_zero_program.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urProgramCreateWithIL(
    ur_context_handle_t Context, ///< [in] handle of the context instance
    const void *IL,              ///< [in] pointer to IL binary.
    size_t Length,               ///< [in] length of `pIL` in bytes.
    const ur_program_properties_t
        *Properties, ///< [in][optional] pointer to program creation properties.
    ur_program_handle_t
        *Program ///< [out] pointer to handle of program object created.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramCreateWithBinary(
    ur_context_handle_t Context, ///< [in] handle of the context instance
    ur_device_handle_t
        Device,            ///< [in] handle to device associated with binary.
    size_t Size,           ///< [in] size in bytes.
    const uint8_t *Binary, ///< [in] pointer to binary.
    const ur_program_properties_t
        *Properties, ///< [in][optional] pointer to program creation properties.
    ur_program_handle_t
        *Program ///< [out] pointer to handle of Program object created.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramBuild(
    ur_context_handle_t Context, ///< [in] handle of the context instance.
    ur_program_handle_t Program, ///< [in] Handle of the program to build.
    const char *Options          ///< [in][optional] pointer to build options
                                 ///< null-terminated string.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramCompile(
    ur_context_handle_t Context, ///< [in] handle of the context instance.
    ur_program_handle_t
        Program,        ///< [in][out] handle of the program to compile.
    const char *Options ///< [in][optional] pointer to build options
                        ///< null-terminated string.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramLink(
    ur_context_handle_t Context, ///< [in] handle of the context instance.
    uint32_t Count, ///< [in] number of program handles in `phPrograms`.
    const ur_program_handle_t *Programs, ///< [in][range(0, count)] pointer to
                                         ///< array of program handles.
    const char *Options, ///< [in][optional] pointer to linker options
                         ///< null-terminated string.
    ur_program_handle_t
        *Program ///< [out] pointer to handle of program object created.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramCreateWithBinary(
    ur_context_handle_t Context, ///< [in] handle of the context instance
    ur_device_handle_t
        Device,            ///< [in] handle to device associated with binary.
    size_t Size,           ///< [in] size in bytes.
    const uint8_t *Binary, ///< [in] pointer to binary.
    ur_program_handle_t
        *Program ///< [out] pointer to handle of Program object created.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramRetain(
    ur_program_handle_t Program ///< [in] handle for the Program to retain
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramRelease(
    ur_program_handle_t Program ///< [in] handle for the Program to release
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramGetFunctionPointer(
    ur_device_handle_t
        Device, ///< [in] handle of the device to retrieve pointer for.
    ur_program_handle_t
        Program, ///< [in] handle of the program to search for function in.
                 ///< The program must already be built to the specified
                 ///< device, or otherwise
                 ///< ::UR_RESULT_ERROR_INVALID_PROGRAM_EXECUTABLE is returned.
    const char *FunctionName, ///< [in] A null-terminates string denoting the
                              ///< mangled function name.
    void **FunctionPointer    ///< [out] Returns the pointer to the function if
                              ///< it is found in the program.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramGetInfo(
    ur_program_handle_t Program, ///< [in] handle of the Program object
    ur_program_info_t PropName,  ///< [in] name of the Program property to query
    size_t PropSize,             ///< [in] the size of the Program property.
    void *ProgramInfo,  ///< [in,out][optional] array of bytes of holding the
                        ///< program info property. If propSize is not equal to
                        ///< or greater than the real number of bytes needed to
                        ///< return the info then the
                        ///< ::UR_RESULT_ERROR_INVALID_SIZE error is returned
                        ///< and pProgramInfo is not used.
    size_t *PropSizeRet ///< [out][optional] pointer to the actual size in
                        ///< bytes of data copied to propName.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramGetBuildInfo(
    ur_program_handle_t Program, ///< [in] handle of the Program object
    ur_device_handle_t Device,   ///< [in] handle of the Device object
    ur_program_build_info_t
        PropName,    ///< [in] name of the Program build info to query
    size_t PropSize, ///< [in] size of the Program build info property.
    void *PropValue, ///< [in,out][optional] value of the Program build
                     ///< property. If propSize is not equal to or greater than
                     ///< the real number of bytes needed to return the info
                     ///< then the ::UR_RESULT_ERROR_INVALID_SIZE error is
                     ///< returned and pKernelInfo is not used.
    size_t *PropSizeRet ///< [out][optional] pointer to the actual size in
                        ///< bytes of data being queried by propName.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramSetSpecializationConstant(
    ur_program_handle_t Program, ///< [in] handle of the Program object
    uint32_t SpecId,             ///< [in] specification constant Id
    size_t SpecSize,      ///< [in] size of the specialization constant value
    const void *SpecValue ///< [in] pointer to the specialization value bytes
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramGetNativeHandle(
    ur_program_handle_t Program,      ///< [in] handle of the program.
    ur_native_handle_t *NativeProgram ///< [out] a pointer to the native
                                      ///< handle of the program.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramCreateWithNativeHandle(
    ur_native_handle_t
        NativeProgram,           ///< [in] the native handle of the program.
    ur_context_handle_t Context, ///< [in] handle of the context instance
    ur_program_handle_t *Program ///< [out] pointer to the handle of the
                                 ///< program object created.
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramSetSpecializationConstants(
    ur_program_handle_t Program, ///< [in] handle of the Program object
    uint32_t Count, ///< [in] the number of elements in the pSpecConstants array
    const ur_specialization_constant_info_t
        *SpecConstants ///< [in][range(0, count)] array of specialization
                       ///< constant value descriptions
) {
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}