//===-- SBCommunication.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBCommunication.h"
#include "SBReproducerPrivate.h"
#include "lldb/API/SBBroadcaster.h"
#include "lldb/Core/Communication.h"
#include "lldb/Host/ConnectionFileDescriptor.h"
#include "lldb/Host/Host.h"
#include "lldb/Utility/Log.h"

using namespace lldb;
using namespace lldb_private;

SBCommunication::SBCommunication() : m_opaque(NULL), m_opaque_owned(false) {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBCommunication);
}

SBCommunication::SBCommunication(const char *broadcaster_name)
    : m_opaque(new Communication(broadcaster_name)), m_opaque_owned(true) {
  LLDB_RECORD_CONSTRUCTOR(SBCommunication, (const char *), broadcaster_name);

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));

  if (log)
    log->Printf("SBCommunication::SBCommunication (broadcaster_name=\"%s\") => "
                "SBCommunication(%p)",
                broadcaster_name, static_cast<void *>(m_opaque));
}

SBCommunication::~SBCommunication() {
  if (m_opaque && m_opaque_owned)
    delete m_opaque;
  m_opaque = NULL;
  m_opaque_owned = false;
}

bool SBCommunication::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBCommunication, IsValid);

  return m_opaque != NULL;
}

bool SBCommunication::GetCloseOnEOF() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBCommunication, GetCloseOnEOF);

  if (m_opaque)
    return m_opaque->GetCloseOnEOF();
  return false;
}

void SBCommunication::SetCloseOnEOF(bool b) {
  LLDB_RECORD_METHOD(void, SBCommunication, SetCloseOnEOF, (bool), b);

  if (m_opaque)
    m_opaque->SetCloseOnEOF(b);
}

ConnectionStatus SBCommunication::Connect(const char *url) {
  LLDB_RECORD_METHOD(lldb::ConnectionStatus, SBCommunication, Connect,
                     (const char *), url);

  if (m_opaque) {
    if (!m_opaque->HasConnection())
      m_opaque->SetConnection(Host::CreateDefaultConnection(url).release());
    return m_opaque->Connect(url, NULL);
  }
  return eConnectionStatusNoConnection;
}

ConnectionStatus SBCommunication::AdoptFileDesriptor(int fd, bool owns_fd) {
  LLDB_RECORD_METHOD(lldb::ConnectionStatus, SBCommunication,
                     AdoptFileDesriptor, (int, bool), fd, owns_fd);

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));

  ConnectionStatus status = eConnectionStatusNoConnection;
  if (m_opaque) {
    if (m_opaque->HasConnection()) {
      if (m_opaque->IsConnected())
        m_opaque->Disconnect();
    }
    m_opaque->SetConnection(new ConnectionFileDescriptor(fd, owns_fd));
    if (m_opaque->IsConnected())
      status = eConnectionStatusSuccess;
    else
      status = eConnectionStatusLostConnection;
  }

  if (log)
    log->Printf(
        "SBCommunication(%p)::AdoptFileDescriptor (fd=%d, ownd_fd=%i) => %s",
        static_cast<void *>(m_opaque), fd, owns_fd,
        Communication::ConnectionStatusAsCString(status));

  return status;
}

ConnectionStatus SBCommunication::Disconnect() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::ConnectionStatus, SBCommunication,
                             Disconnect);

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));

  ConnectionStatus status = eConnectionStatusNoConnection;
  if (m_opaque)
    status = m_opaque->Disconnect();

  if (log)
    log->Printf("SBCommunication(%p)::Disconnect () => %s",
                static_cast<void *>(m_opaque),
                Communication::ConnectionStatusAsCString(status));

  return status;
}

bool SBCommunication::IsConnected() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBCommunication, IsConnected);

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));
  bool result = false;
  if (m_opaque)
    result = m_opaque->IsConnected();

  if (log)
    log->Printf("SBCommunication(%p)::IsConnected () => %i",
                static_cast<void *>(m_opaque), result);

  return false;
}

size_t SBCommunication::Read(void *dst, size_t dst_len, uint32_t timeout_usec,
                             ConnectionStatus &status) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));
  if (log)
    log->Printf("SBCommunication(%p)::Read (dst=%p, dst_len=%" PRIu64
                ", timeout_usec=%u, &status)...",
                static_cast<void *>(m_opaque), static_cast<void *>(dst),
                static_cast<uint64_t>(dst_len), timeout_usec);
  size_t bytes_read = 0;
  Timeout<std::micro> timeout = timeout_usec == UINT32_MAX
                                    ? Timeout<std::micro>(llvm::None)
                                    : std::chrono::microseconds(timeout_usec);
  if (m_opaque)
    bytes_read = m_opaque->Read(dst, dst_len, timeout, status, NULL);
  else
    status = eConnectionStatusNoConnection;

  if (log)
    log->Printf("SBCommunication(%p)::Read (dst=%p, dst_len=%" PRIu64
                ", timeout_usec=%u, &status=%s) => %" PRIu64,
                static_cast<void *>(m_opaque), static_cast<void *>(dst),
                static_cast<uint64_t>(dst_len), timeout_usec,
                Communication::ConnectionStatusAsCString(status),
                static_cast<uint64_t>(bytes_read));
  return bytes_read;
}

size_t SBCommunication::Write(const void *src, size_t src_len,
                              ConnectionStatus &status) {
  size_t bytes_written = 0;
  if (m_opaque)
    bytes_written = m_opaque->Write(src, src_len, status, NULL);
  else
    status = eConnectionStatusNoConnection;

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));
  if (log)
    log->Printf("SBCommunication(%p)::Write (src=%p, src_len=%" PRIu64
                ", &status=%s) => %" PRIu64,
                static_cast<void *>(m_opaque), static_cast<const void *>(src),
                static_cast<uint64_t>(src_len),
                Communication::ConnectionStatusAsCString(status),
                static_cast<uint64_t>(bytes_written));

  return 0;
}

bool SBCommunication::ReadThreadStart() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBCommunication, ReadThreadStart);

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));

  bool success = false;
  if (m_opaque)
    success = m_opaque->StartReadThread();

  if (log)
    log->Printf("SBCommunication(%p)::ReadThreadStart () => %i",
                static_cast<void *>(m_opaque), success);

  return success;
}

bool SBCommunication::ReadThreadStop() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBCommunication, ReadThreadStop);

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));
  if (log)
    log->Printf("SBCommunication(%p)::ReadThreadStop ()...",
                static_cast<void *>(m_opaque));

  bool success = false;
  if (m_opaque)
    success = m_opaque->StopReadThread();

  if (log)
    log->Printf("SBCommunication(%p)::ReadThreadStop () => %i",
                static_cast<void *>(m_opaque), success);

  return success;
}

bool SBCommunication::ReadThreadIsRunning() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBCommunication, ReadThreadIsRunning);

  bool result = false;
  if (m_opaque)
    result = m_opaque->ReadThreadIsRunning();
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));
  if (log)
    log->Printf("SBCommunication(%p)::ReadThreadIsRunning () => %i",
                static_cast<void *>(m_opaque), result);
  return result;
}

bool SBCommunication::SetReadThreadBytesReceivedCallback(
    ReadThreadBytesReceived callback, void *callback_baton) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));

  bool result = false;
  if (m_opaque) {
    m_opaque->SetReadThreadBytesReceivedCallback(callback, callback_baton);
    result = true;
  }

  if (log)
    log->Printf("SBCommunication(%p)::SetReadThreadBytesReceivedCallback "
                "(callback=%p, baton=%p) => %i",
                static_cast<void *>(m_opaque),
                reinterpret_cast<void *>(reinterpret_cast<intptr_t>(callback)),
                static_cast<void *>(callback_baton), result);

  return result;
}

SBBroadcaster SBCommunication::GetBroadcaster() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBBroadcaster, SBCommunication,
                             GetBroadcaster);

  SBBroadcaster broadcaster(m_opaque, false);

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));

  if (log)
    log->Printf("SBCommunication(%p)::GetBroadcaster () => SBBroadcaster (%p)",
                static_cast<void *>(m_opaque),
                static_cast<void *>(broadcaster.get()));

  return LLDB_RECORD_RESULT(broadcaster);
}

const char *SBCommunication::GetBroadcasterClass() {
  LLDB_RECORD_STATIC_METHOD_NO_ARGS(const char *, SBCommunication,
                                    GetBroadcasterClass);

  return Communication::GetStaticBroadcasterClass().AsCString();
}

//
// void
// SBCommunication::CreateIfNeeded ()
//{
//    if (m_opaque == NULL)
//    {
//        static uint32_t g_broadcaster_num;
//        char broadcaster_name[256];
//        ::snprintf (name, broadcaster_name, "%p SBCommunication", this);
//        m_opaque = new Communication (broadcaster_name);
//        m_opaque_owned = true;
//    }
//    assert (m_opaque);
//}
//
//
