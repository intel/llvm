/*
 *
 * Copyright (C) 2022-2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * @file ur_object.h
 *
 */

#ifndef UR_OBJECT_H
#define UR_OBJECT_H 1

#include "ur_singleton.h"
#include "ur_util.h"

//////////////////////////////////////////////////////////////////////////
struct dditable_t
{
    ur_dditable_t   ur;
    //urs_dditable_t  urs;
    //urt_dditable_t  urt;
};

//////////////////////////////////////////////////////////////////////////
template<typename _handle_t>
class __urdlllocal object_t
{
public:
    using handle_t = _handle_t;

    handle_t    handle;
    dditable_t* dditable;

    object_t() = delete;

    object_t( handle_t _handle, dditable_t* _dditable )
        : handle( _handle ), dditable( _dditable )
    {
    }

    ~object_t() = default;
};

#endif /* UR_OBJECT_H */
