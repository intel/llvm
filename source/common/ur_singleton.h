/*
 *
 * Copyright (C) 2022-2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef UR_SINGLETON_H
#define UR_SINGLETON_H 1

#include <memory>
#include <mutex>
#include <unordered_map>

//////////////////////////////////////////////////////////////////////////
/// a abstract factory for creation of singleton objects
template <typename singleton_tn, typename key_tn> class singleton_factory_t {
  protected:
    using singleton_t = singleton_tn;
    using key_t = typename std::conditional<std::is_pointer<key_tn>::value,
                                            size_t, key_tn>::type;

    using ptr_t = std::unique_ptr<singleton_t>;
    using map_t = std::unordered_map<key_t, ptr_t>;

    std::mutex mut; ///< lock for thread-safety
    map_t map;      ///< single instance of singleton for each unique key

    //////////////////////////////////////////////////////////////////////////
    /// extract the key from parameter list and if necessary, convert type
    template <typename... Ts> key_t getKey(key_tn key, Ts &&... params) {
        return reinterpret_cast<key_t>(key);
    }

  public:
    //////////////////////////////////////////////////////////////////////////
    /// default ctor/dtor
    singleton_factory_t() = default;
    ~singleton_factory_t() = default;

    //////////////////////////////////////////////////////////////////////////
    /// gets a pointer to a unique instance of singleton
    /// if no instance exists, then creates a new instance
    /// the params are forwarded to the ctor of the singleton
    /// the first parameter must be the unique identifier of the instance
    template <typename... Ts> singleton_tn *getInstance(Ts &&... params) {
        auto key = getKey(std::forward<Ts>(params)...);

        if (key == 0) { // No zero keys allowed in map
            return static_cast<singleton_tn *>(0);
        }

        std::lock_guard<std::mutex> lk(mut);
        auto iter = map.find(key);

        if (map.end() == iter) {
            auto ptr =
                std::make_unique<singleton_t>(std::forward<Ts>(params)...);
            iter = map.emplace(key, std::move(ptr)).first;
        }
        return iter->second.get();
    }

    //////////////////////////////////////////////////////////////////////////
    /// once the key is no longer valid, release the singleton
    void release(key_tn key) {
        std::lock_guard<std::mutex> lk(mut);
        map.erase(getKey(key));
    }
};

#endif /* UR_SINGLETON_H */
