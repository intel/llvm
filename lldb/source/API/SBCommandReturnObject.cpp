//===-- SBCommandReturnObject.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBCommandReturnObject.h"
#include "SBReproducerPrivate.h"
#include "Utils.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBStream.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Status.h"

using namespace lldb;
using namespace lldb_private;

SBCommandReturnObject::SBCommandReturnObject()
    : m_opaque_up(new CommandReturnObject()) {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBCommandReturnObject);
}

SBCommandReturnObject::SBCommandReturnObject(const SBCommandReturnObject &rhs)
    : m_opaque_up() {
  LLDB_RECORD_CONSTRUCTOR(SBCommandReturnObject,
                          (const lldb::SBCommandReturnObject &), rhs);

  m_opaque_up = clone(rhs.m_opaque_up);
}

SBCommandReturnObject::SBCommandReturnObject(CommandReturnObject *ptr)
    : m_opaque_up(ptr) {
  LLDB_RECORD_CONSTRUCTOR(SBCommandReturnObject,
                          (lldb_private::CommandReturnObject *), ptr);
}

SBCommandReturnObject::~SBCommandReturnObject() = default;

CommandReturnObject *SBCommandReturnObject::Release() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb_private::CommandReturnObject *,
                             SBCommandReturnObject, Release);

  return m_opaque_up.release();
}

const SBCommandReturnObject &SBCommandReturnObject::
operator=(const SBCommandReturnObject &rhs) {
  LLDB_RECORD_METHOD(
      const lldb::SBCommandReturnObject &,
      SBCommandReturnObject, operator=,(const lldb::SBCommandReturnObject &),
      rhs);

  if (this != &rhs)
    m_opaque_up = clone(rhs.m_opaque_up);
  return *this;
}

bool SBCommandReturnObject::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBCommandReturnObject, IsValid);

  return m_opaque_up != nullptr;
}

const char *SBCommandReturnObject::GetOutput() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBCommandReturnObject, GetOutput);

  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));

  if (m_opaque_up) {
    llvm::StringRef output = m_opaque_up->GetOutputData();
    ConstString result(output.empty() ? llvm::StringRef("") : output);

    if (log)
      log->Printf("SBCommandReturnObject(%p)::GetOutput () => \"%s\"",
                  static_cast<void *>(m_opaque_up.get()), result.AsCString());

    return result.AsCString();
  }

  if (log)
    log->Printf("SBCommandReturnObject(%p)::GetOutput () => nullptr",
                static_cast<void *>(m_opaque_up.get()));

  return nullptr;
}

const char *SBCommandReturnObject::GetError() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBCommandReturnObject, GetError);

  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));

  if (m_opaque_up) {
    llvm::StringRef output = m_opaque_up->GetErrorData();
    ConstString result(output.empty() ? llvm::StringRef("") : output);
    if (log)
      log->Printf("SBCommandReturnObject(%p)::GetError () => \"%s\"",
                  static_cast<void *>(m_opaque_up.get()), result.AsCString());

    return result.AsCString();
  }

  if (log)
    log->Printf("SBCommandReturnObject(%p)::GetError () => nullptr",
                static_cast<void *>(m_opaque_up.get()));

  return nullptr;
}

size_t SBCommandReturnObject::GetOutputSize() {
  LLDB_RECORD_METHOD_NO_ARGS(size_t, SBCommandReturnObject, GetOutputSize);

  return (m_opaque_up ? m_opaque_up->GetOutputData().size() : 0);
}

size_t SBCommandReturnObject::GetErrorSize() {
  LLDB_RECORD_METHOD_NO_ARGS(size_t, SBCommandReturnObject, GetErrorSize);

  return (m_opaque_up ? m_opaque_up->GetErrorData().size() : 0);
}

size_t SBCommandReturnObject::PutOutput(FILE *fh) {
  LLDB_RECORD_METHOD(size_t, SBCommandReturnObject, PutOutput, (FILE *), fh);

  if (fh) {
    size_t num_bytes = GetOutputSize();
    if (num_bytes)
      return ::fprintf(fh, "%s", GetOutput());
  }
  return 0;
}

size_t SBCommandReturnObject::PutError(FILE *fh) {
  LLDB_RECORD_METHOD(size_t, SBCommandReturnObject, PutError, (FILE *), fh);

  if (fh) {
    size_t num_bytes = GetErrorSize();
    if (num_bytes)
      return ::fprintf(fh, "%s", GetError());
  }
  return 0;
}

void SBCommandReturnObject::Clear() {
  LLDB_RECORD_METHOD_NO_ARGS(void, SBCommandReturnObject, Clear);

  if (m_opaque_up)
    m_opaque_up->Clear();
}

lldb::ReturnStatus SBCommandReturnObject::GetStatus() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::ReturnStatus, SBCommandReturnObject,
                             GetStatus);

  return (m_opaque_up ? m_opaque_up->GetStatus() : lldb::eReturnStatusInvalid);
}

void SBCommandReturnObject::SetStatus(lldb::ReturnStatus status) {
  LLDB_RECORD_METHOD(void, SBCommandReturnObject, SetStatus,
                     (lldb::ReturnStatus), status);

  if (m_opaque_up)
    m_opaque_up->SetStatus(status);
}

bool SBCommandReturnObject::Succeeded() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBCommandReturnObject, Succeeded);

  return (m_opaque_up ? m_opaque_up->Succeeded() : false);
}

bool SBCommandReturnObject::HasResult() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBCommandReturnObject, HasResult);

  return (m_opaque_up ? m_opaque_up->HasResult() : false);
}

void SBCommandReturnObject::AppendMessage(const char *message) {
  LLDB_RECORD_METHOD(void, SBCommandReturnObject, AppendMessage, (const char *),
                     message);

  if (m_opaque_up)
    m_opaque_up->AppendMessage(message);
}

void SBCommandReturnObject::AppendWarning(const char *message) {
  LLDB_RECORD_METHOD(void, SBCommandReturnObject, AppendWarning, (const char *),
                     message);

  if (m_opaque_up)
    m_opaque_up->AppendWarning(message);
}

CommandReturnObject *SBCommandReturnObject::operator->() const {
  return m_opaque_up.get();
}

CommandReturnObject *SBCommandReturnObject::get() const {
  return m_opaque_up.get();
}

CommandReturnObject &SBCommandReturnObject::operator*() const {
  assert(m_opaque_up.get());
  return *(m_opaque_up.get());
}

CommandReturnObject &SBCommandReturnObject::ref() const {
  assert(m_opaque_up.get());
  return *(m_opaque_up.get());
}

void SBCommandReturnObject::SetLLDBObjectPtr(CommandReturnObject *ptr) {
  if (m_opaque_up)
    m_opaque_up.reset(ptr);
}

bool SBCommandReturnObject::GetDescription(SBStream &description) {
  LLDB_RECORD_METHOD(bool, SBCommandReturnObject, GetDescription,
                     (lldb::SBStream &), description);

  Stream &strm = description.ref();

  if (m_opaque_up) {
    description.Printf("Error:  ");
    lldb::ReturnStatus status = m_opaque_up->GetStatus();
    if (status == lldb::eReturnStatusStarted)
      strm.PutCString("Started");
    else if (status == lldb::eReturnStatusInvalid)
      strm.PutCString("Invalid");
    else if (m_opaque_up->Succeeded())
      strm.PutCString("Success");
    else
      strm.PutCString("Fail");

    if (GetOutputSize() > 0)
      strm.Printf("\nOutput Message:\n%s", GetOutput());

    if (GetErrorSize() > 0)
      strm.Printf("\nError Message:\n%s", GetError());
  } else
    strm.PutCString("No value");

  return true;
}

void SBCommandReturnObject::SetImmediateOutputFile(FILE *fh) {
  LLDB_RECORD_METHOD(void, SBCommandReturnObject, SetImmediateOutputFile,
                     (FILE *), fh);

  SetImmediateOutputFile(fh, false);
}

void SBCommandReturnObject::SetImmediateErrorFile(FILE *fh) {
  LLDB_RECORD_METHOD(void, SBCommandReturnObject, SetImmediateErrorFile,
                     (FILE *), fh);

  SetImmediateErrorFile(fh, false);
}

void SBCommandReturnObject::SetImmediateOutputFile(FILE *fh,
                                                   bool transfer_ownership) {
  LLDB_RECORD_METHOD(void, SBCommandReturnObject, SetImmediateOutputFile,
                     (FILE *, bool), fh, transfer_ownership);

  if (m_opaque_up)
    m_opaque_up->SetImmediateOutputFile(fh, transfer_ownership);
}

void SBCommandReturnObject::SetImmediateErrorFile(FILE *fh,
                                                  bool transfer_ownership) {
  LLDB_RECORD_METHOD(void, SBCommandReturnObject, SetImmediateErrorFile,
                     (FILE *, bool), fh, transfer_ownership);

  if (m_opaque_up)
    m_opaque_up->SetImmediateErrorFile(fh, transfer_ownership);
}

void SBCommandReturnObject::PutCString(const char *string, int len) {
  LLDB_RECORD_METHOD(void, SBCommandReturnObject, PutCString,
                     (const char *, int), string, len);

  if (m_opaque_up) {
    if (len == 0 || string == nullptr || *string == 0) {
      return;
    } else if (len > 0) {
      std::string buffer(string, len);
      m_opaque_up->AppendMessage(buffer.c_str());
    } else
      m_opaque_up->AppendMessage(string);
  }
}

const char *SBCommandReturnObject::GetOutput(bool only_if_no_immediate) {
  LLDB_RECORD_METHOD(const char *, SBCommandReturnObject, GetOutput, (bool),
                     only_if_no_immediate);

  if (!m_opaque_up)
    return nullptr;
  if (!only_if_no_immediate ||
      m_opaque_up->GetImmediateOutputStream().get() == nullptr)
    return GetOutput();
  return nullptr;
}

const char *SBCommandReturnObject::GetError(bool only_if_no_immediate) {
  LLDB_RECORD_METHOD(const char *, SBCommandReturnObject, GetError, (bool),
                     only_if_no_immediate);

  if (!m_opaque_up)
    return nullptr;
  if (!only_if_no_immediate ||
      m_opaque_up->GetImmediateErrorStream().get() == nullptr)
    return GetError();
  return nullptr;
}

size_t SBCommandReturnObject::Printf(const char *format, ...) {
  if (m_opaque_up) {
    va_list args;
    va_start(args, format);
    size_t result = m_opaque_up->GetOutputStream().PrintfVarArg(format, args);
    va_end(args);
    return result;
  }
  return 0;
}

void SBCommandReturnObject::SetError(lldb::SBError &error,
                                     const char *fallback_error_cstr) {
  LLDB_RECORD_METHOD(void, SBCommandReturnObject, SetError,
                     (lldb::SBError &, const char *), error,
                     fallback_error_cstr);

  if (m_opaque_up) {
    if (error.IsValid())
      m_opaque_up->SetError(error.ref(), fallback_error_cstr);
    else if (fallback_error_cstr)
      m_opaque_up->SetError(Status(), fallback_error_cstr);
  }
}

void SBCommandReturnObject::SetError(const char *error_cstr) {
  LLDB_RECORD_METHOD(void, SBCommandReturnObject, SetError, (const char *),
                     error_cstr);

  if (m_opaque_up && error_cstr)
    m_opaque_up->SetError(error_cstr);
}
