// emhash8::HashSet for C++11
// version 1.3.2
// https://github.com/ktprime/ktprime/blob/master/hash_set.hpp
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
// SPDX-License-Identifier: MIT
// Copyright (c) 2019-2022 Huang Yuanbing & bailuzhou AT 163.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE


// From
// NUMBER OF PROBES / LOOKUP       Successful            Unsuccessful
// Quadratic collision resolution   1 - ln(1-L) - L/2    1/(1-L) - L - ln(1-L)
// Linear collision resolution     [1+1/(1-L)]/2         [1+1/(1-L)2]/2
//
// -- enlarge_factor --           0.10  0.50  0.60  0.75  0.80  0.90  0.99
// QUADRATIC COLLISION RES.
//    probes/successful lookup    1.05  1.44  1.62  2.01  2.21  2.85  5.11
//    probes/unsuccessful lookup  1.11  2.19  2.82  4.64  5.81  11.4  103.6
// LINEAR COLLISION RES.
//    probes/successful lookup    1.06  1.5   1.75  2.5   3.0   5.5   50.5
//    probes/unsuccessful lookup  1.12  2.5   3.6   8.5   13.0  50.0

#pragma once

#include <cstring>
#include <string>
#include <cstdlib>
#include <type_traits>
#include <cassert>
#include <utility>
#include <cstdint>
#include <functional>
#include <iterator>

#ifdef __has_include
    #if __has_include("wyhash.h")
    #include "wyhash.h"
    #endif
#elif EMH_WY_HASH
    #include "wyhash.h"
#endif

#ifdef EMH_ENTRY
    #undef EMH_ENTRY
#endif

// likely/unlikely
#if defined(__GNUC__) || defined(__INTEL_COMPILER) || defined(__clang__)
#    define EMH_LIKELY(condition) __builtin_expect(condition, 1)
#    define EMH_UNLIKELY(condition) __builtin_expect(condition, 0)
#else
#    define EMH_LIKELY(condition) condition
#    define EMH_UNLIKELY(condition) condition
#endif

#define EMH_ENTRY(key, bucket) new(_pairs + bucket) PairT(key, bucket), _num_filled ++
#define EMH_MOVE(bucket, bobj) new(_pairs + bucket) PairT(bobj)

namespace emhash2 {

/// A cache-friendly hash table with open addressing, linear probing and power-of-two capacity
template <typename KeyT, typename HashT = std::hash<KeyT>, typename EqT = std::equal_to<KeyT>>
class HashSet
{
public:
    //if constexpr (sizeof(KeyT) <= 4 && std::is_integral<KeyT>::value)
#ifndef EMH_SIZE_TYPE_64BIT
        typedef uint32_t size_type;
        static constexpr size_type INACTIVE = 0-1u;
#else
        typedef uint64_t size_type;
        static constexpr size_type INACTIVE = 0-1ull;
#endif

    typedef HashSet<KeyT, HashT, EqT> htype;
    typedef std::pair<KeyT, size_type> PairT;
    static constexpr bool bInCacheLine = sizeof(PairT) < 64 * 2 / 3;
    static constexpr float default_load_factor = 0.95f;

    typedef KeyT     value_type;
    typedef KeyT&    reference;
    typedef KeyT*    pointer;
    typedef const KeyT& const_reference;

    class iterator
    {
    public:
        typedef std::forward_iterator_tag iterator_category;
        typedef std::ptrdiff_t            difference_type;
        typedef KeyT                      value_type;
        typedef value_type*               pointer;
        typedef value_type&               reference;

        iterator() { }
        iterator(htype* hash_set, size_type bucket) : _set(hash_set), _bucket(bucket) { }

        iterator& operator++()
        {
            goto_next_element();
            return *this;
        }

        iterator operator++(int)
        {
            auto old_index = _bucket;
            goto_next_element();
            return {_set, old_index};
        }

        reference operator*() const
        {
            return _set->_pairs[_bucket].first;
        }

        pointer operator->() const
        {
            return &(_set->_pairs[_bucket].first);
        }

        bool operator==(const iterator& rhs) const
        {
            return _bucket == rhs._bucket;
        }

        bool operator!=(const iterator& rhs) const
        {
            return _bucket != rhs._bucket;
        }

        size_type bucket() const
        {
            return _bucket;
        }

    private:
        void goto_next_element()
        {
            do {
                _bucket++;
            } while (_set->_pairs[_bucket].second == INACTIVE);
        }

    public:
        htype* _set;
        size_type _bucket;
    };

    class const_iterator
    {
    public:
        typedef std::forward_iterator_tag iterator_category;
        typedef std::ptrdiff_t            difference_type;
        typedef KeyT                      value_type;
        typedef value_type*               pointer;
        typedef value_type&               reference;

        const_iterator() { }
        const_iterator(iterator proto) : _set(proto._set), _bucket(proto._bucket) {  }
        const_iterator(const htype* hash_set, size_type bucket) : _set(hash_set), _bucket(bucket) {  }

        const_iterator& operator++()
        {
            goto_next_element();
            return *this;
        }

        const_iterator operator++(int)
        {
            auto old_index = _bucket;
            goto_next_element();
            return {_set, old_index};
        }

        reference operator*() const
        {
            return _set->_pairs[_bucket].first;
        }

        pointer operator->() const
        {
            return &(_set->_pairs[_bucket].first);
        }

        bool operator==(const const_iterator& rhs) const
        {
            return _bucket == rhs._bucket;
        }

        bool operator!=(const const_iterator& rhs) const
        {
            return _bucket != rhs._bucket;
        }

        size_type bucket() const
        {
            return _bucket;
        }

    private:
        void goto_next_element()
        {
            do {
                _bucket++;
            } while (_set->_pairs[_bucket].second == INACTIVE);
        }

    public:
        const htype* _set;
        size_type  _bucket;
    };

    // ------------------------------------------------------------------------

    void init(size_type bucket, float lf)
    {
        _pairs = nullptr;
        _num_buckets = _num_filled = 0;

        max_load_factor(lf);
        reserve(bucket);
    }

    HashSet(size_type bucket = 2, float lf = default_load_factor)
    {
        init(bucket, lf);
    }

    HashSet(const HashSet& other)
    {
        _pairs = alloc_bucket(other._num_buckets);
        clone(other);
    }

    HashSet(HashSet&& other)
    {
#ifdef EMH_MOVE_EMPTY
        _pairs = nullptr;
        _num_buckets = _num_filled = 0;
#else
        init(4, default_load_factor);
#endif
        swap(other);
    }

    HashSet(std::initializer_list<value_type> il, size_type n = 8)
    {
        init(std::max((size_type)il.size(), n));
        for (auto begin = il.begin(); begin != il.end(); ++begin)
            insert(*begin);
    }

    template<class InputIt>
    HashSet(InputIt first, InputIt last, size_type bucket_count=4)
    {
        init(std::distance(first, last) + bucket_count, default_load_factor);
        for (; first != last; ++first)
            emplace(*first);
    }

    HashSet& operator=(const HashSet& other)
    {
        if (this == &other)
            return *this;

        if (!std::is_trivially_destructible<KeyT>::value)
            clearkv();

        if (_num_buckets != other._num_buckets) {
            free(_pairs);
            _pairs = alloc_bucket(other._num_buckets);
        }

        clone(other);
        return *this;
    }

    HashSet& operator=(HashSet&& other)
    {
        if (this != &other) {
            swap(other);
            other.clear();
        }
        return *this;
    }

    ~HashSet()
    {
        if (!std::is_trivially_destructible<KeyT>::value)
            clearkv();

        free(_pairs);
    }

    void clone(const HashSet& other)
    {
        _hasher      = other._hasher;
        _num_buckets = other._num_buckets;
        _num_filled  = other._num_filled;
        _last_colls  = other._last_colls;
        _mask        = other._mask;
        _loadlf      = other._loadlf;

        if (std::is_trivially_copyable<KeyT>::value) {
            memcpy(_pairs, other._pairs, _num_buckets * sizeof(PairT));
        } else {
            auto old_pairs = other._pairs;
            for (size_type bucket = 0; bucket < _num_buckets; bucket++) {
                auto next_bucket = _pairs[bucket].second = old_pairs[bucket].second;
                if (next_bucket != INACTIVE)
                    EMH_MOVE(bucket, old_pairs[bucket]);
            }
        }
        _pairs[_num_buckets].second = _pairs[_num_buckets + 1].second = 0;
    }

    void swap(HashSet& other)
    {
        std::swap(_hasher, other._hasher);
//      std::swap(_eq, other._eq);
        std::swap(_pairs, other._pairs);
        std::swap(_num_buckets, other._num_buckets);
        std::swap(_num_filled, other._num_filled);
        std::swap(_mask, other._mask);
        std::swap(_loadlf, other._loadlf);
        std::swap(_last_colls, other._last_colls);
    }

    // -------------------------------------------------------------

    iterator begin()
    {
        size_type bucket = 0;
        while (_pairs[bucket].second == INACTIVE) {
            ++bucket;
        }
        return {this, bucket};
    }

    const_iterator cbegin() const
    {
        size_type bucket = 0;
        while (_pairs[bucket].second == INACTIVE) {
            ++bucket;
        }
        return {this, bucket};
    }

    const_iterator begin() const
    {
        return cbegin();
    }

    iterator end()
    {
        return {this, _num_buckets};
    }

    const_iterator cend() const
    {
        return {this, _num_buckets};
    }

    const_iterator end() const
    {
        return cend();
    }

    size_type size() const
    {
        return _num_filled;
    }

    bool empty() const
    {
        return _num_filled == 0;
    }

    // Returns the number of buckets.
    size_type bucket_count() const
    {
        return _num_buckets;
    }

    /// Returns average number of elements per bucket.
    float load_factor() const
    {
        return static_cast<float>(_num_filled) / (_num_buckets + 0.01f);
    }

    HashT& hash_function() const
    {
        return _hasher;
    }

    EqT& key_eq() const
    {
        return _eq;
    }

    constexpr float max_load_factor() const
    {
        return (1 << 27) / (float)_loadlf;
    }

    void max_load_factor(float value)
    {
        if (value < 0.999 && value > 0.2f)
            _loadlf = (1 << 27) / value;
    }

    constexpr size_type max_size() const
    {
        return (1u << 31) / sizeof(PairT);
    }

    constexpr size_type max_bucket_count() const
    {
        return (1u << 31) / sizeof(PairT);
    }

#ifndef TEST_TIMER_FEATURE
    int64_t fast_search(int64_t key, size_type buckets) const
    {
        auto min_key = key + buckets - 1;
        auto bfrom = key;

        while(buckets --)
        {
            const auto bucket = bfrom ++ & _mask;
            const auto next_bucket = _pairs[bucket].second;
            if (next_bucket == INACTIVE) {
                key++;
                continue;
            }

            const auto node = _pairs[bucket].first;
//            if (bucket == hash_bucket(node))
//                key++;

            if (node->expire < min_key && node->expire > 0) {
                min_key = node->expire;
//                if (min_key <= key) break;
            }
        }
        return min_key;
    }

    int64_t near_bucket(int64_t key, size_type buckets) const
    {
        auto bfrom = get_main_bucket(key);
        if (bfrom == -1)
            bfrom = key;

        while (buckets--) {
            const auto ckey   = key++;
            const auto bucket = bfrom++ & _mask;
            auto next_bucket  = _pairs[bucket].second;
            if (next_bucket == INACTIVE)
                continue;
            else if (_pairs[bucket].first->expire == ckey)
                break;

            const auto& node = _pairs[next_bucket].first;
            //* check current bucket_key is main bucket or not
            const auto main_bucket = hash_bucket(node);
            if (main_bucket != bucket)
                continue;
            else if (node->expire == ckey)
                break;
            //search from main bucket
            next_bucket = _pairs[main_bucket].second;
            if (next_bucket == main_bucket)
                continue;

            while (true) {
                const auto nbucket = _pairs[next_bucket].second;
                if (_pairs[next_bucket].first->expire == ckey)
                    break;
                if (nbucket == next_bucket)
                    break;
                next_bucket = nbucket;
            }
        }

        return key - 1;
    }

#if 0
    size_type get_bucket_value(const size_type main_bucket, const int64_t key, std::vector<KeyT>& vec)
    {
        auto node = _pairs[main_bucket].first;
        assert(main_bucket == hash_bucket(node));
        if (node->expire <= key)
            vec.push_back(node);

        auto next_bucket = _pairs[main_bucket].second;
        if (next_bucket == main_bucket) {
            //clear_bucket(main_bucket);
            return 1;
        }

        while (true) {
            const auto nbucket = _pairs[next_bucket].second;

            node = _pairs[next_bucket].first;
            if (node->expire <= key)
                vec.push_back(node);

            if (nbucket == next_bucket)
                break;
            next_bucket = nbucket;
        }
        return vec.size();
    }
#endif

    size_type get_main_bucket(const int64_t key) const
    {
        const auto bucket = key & _mask;
        const auto next_bucket = _pairs[bucket].second;
        if (next_bucket == INACTIVE)
            return -1;

        const auto& node = _pairs[bucket].first;
        //check current bucket_key is in main bucket or not
        const auto main_bucket = hash_bucket(node);
        //assert(main_bucket == hash_bucket(_pairs[next_bucket].first));
        if (main_bucket != bucket)
            return -1;
        else if (next_bucket == main_bucket && node->expire != key)
            return -1;

        return main_bucket;
    }
#endif

#ifdef EMH_STATIS
    size_type bucket_main() const
    {
        auto bucket_size = 0;
        for (size_type bucket = 0; bucket < _num_buckets; ++bucket)
            bucket_size += _pairs[bucket].second == bucket;
        return bucket_size;
    }

    //Returns the bucket number where the element with key hashed is located.
    size_type get_main_bucket2(const KeyT& key) const
    {
        const auto bucket = hash_bucket(key);
        const auto next_bucket = _pairs[bucket].second;
        if (next_bucket == INACTIVE)
            return INACTIVE;

        return hash_bucket(_pairs[bucket].first);
    }

    size_type get_main_bucket(const size_type bucket) const
    {
        auto next_bucket = _pairs[bucket].second;
        if (next_bucket == INACTIVE)
            return INACTIVE;

        return hash_bucket(_pairs[bucket].first);
    }

    //Returns the number of elements in bucket n.
    size_type bucket_size(const size_type bucket) const
    {
        auto next_bucket = _pairs[bucket].second;
        if (next_bucket == INACTIVE)
            return 0;

        next_bucket = hash_bucket(_pairs[bucket].first);
        size_type bucket_size = 1;

        //iterator each item in current main bucket
        while (true) {
            const auto nbucket = _pairs[next_bucket].second;
            if (nbucket == next_bucket) {
                break;
            }
            bucket_size++;
            next_bucket = nbucket;
        }
        return bucket_size;
    }

    int get_cache_info(size_type bucket, size_type next_bucket) const
    {
        auto pbucket = reinterpret_cast<size_t>(&_pairs[bucket]);
        auto pnext   = reinterpret_cast<size_t>(&_pairs[next_bucket]);
        if (pbucket / 64 == pnext / 64)
            return 0;
        auto diff = pbucket > pnext ? (pbucket - pnext) : pnext - pbucket;
        if (diff < 127 * 64)
            return diff / 64 + 1;
        return 127;
    }

    int get_bucket_info(const size_type bucket, size_type steps[], const size_type slots) const
    {
        auto next_bucket = _pairs[bucket].second;
        if (next_bucket == INACTIVE)
            return -1;

        const auto& bucket_key = _pairs[bucket].first;
        const auto main_bucket = hash_bucket(bucket_key);
        if (main_bucket != bucket)
            return 0;
        else if (next_bucket == bucket)
            return 1;

        steps[get_cache_info(bucket, next_bucket) % slots] ++;
        size_type ibucket_size = 2;
        //find a new empty and linked it to tail
        while (true) {
            const auto nbucket = _pairs[next_bucket].second;
            if (nbucket == next_bucket)
                break;

            steps[get_cache_info(nbucket, next_bucket) % slots] ++;
            ibucket_size ++;
            next_bucket = nbucket;
        }
        return ibucket_size;
    }

    void dump_statics() const
    {
        size_type buckets[129] = {0};
        size_type steps[129]   = {0};
        for (size_type bucket = 0; bucket < _num_buckets; ++bucket) {
            auto bsize = get_bucket_info(bucket, steps, 128);
            if (bsize > 0)
                buckets[bsize] ++;
        }

        size_type sumb = 0, collision = 0, sumc = 0, finds = 0, sumn = 0;
        puts("============== buckets size ration =========");
        for (size_type i = 0; i < sizeof(buckets) / sizeof(buckets[0]); i++) {
            const auto bucketsi = buckets[i];
            if (bucketsi == 0)
                continue;
            sumb += bucketsi;
            sumn += bucketsi * i;
            collision += bucketsi * (i - 1);
            finds += bucketsi * i * (i + 1) / 2;
            printf("  %2u  %8u  %.2lf  %.2lf\n", i, bucketsi, bucketsi * 100.0 * i / _num_filled, sumn * 100.0 / _num_filled);
        }

        puts("========== collision miss ration ===========");
        for (size_type i = 0; i < sizeof(steps) / sizeof(steps[0]); i++) {
            sumc += steps[i];
            if (steps[i] <= 2)
                continue;
            printf("  %2u  %8u  %.2lf  %.2lf\n", i, steps[i], steps[i] * 100.0 / collision, sumc * 100.0 / collision);
        }

        if (sumb == 0)  return;
        printf("    _num_filled/bucket_size/packed collision/cache_miss/hit_find = %u/%.2lf/%zd/ %.2lf%%/%.2lf%%/%.2lf\n",
                _num_filled, _num_filled * 1.0 / sumb, sizeof(PairT), (collision * 100.0 / _num_filled), (collision - steps[0]) * 100.0 / _num_filled, finds * 1.0 / _num_filled);
        assert(sumn == _num_filled);
        assert(sumc == collision);
    }
#endif

    // ------------------------------------------------------------

    iterator find(const KeyT& key)
    {
        return {this, find_filled_bucket(key)};
    }

    const_iterator find(const KeyT& key) const
    {
        return {this, find_filled_bucket(key)};
    }

    bool contains(const KeyT& key) const
    {
        return find_filled_bucket(key) != _num_buckets;
    }

    size_type count(const KeyT& key) const
    {
        return find_filled_bucket(key) == _num_buckets ? 0 : 1;
    }

    /// Returns a pair consisting of an iterator to the inserted element
    /// (or to the element that prevented the insertion)
    /// and a bool denoting whether the insertion took place.
    std::pair<iterator, bool> insert(const KeyT& key)
    {
        check_expand_need();
        const auto bucket = find_or_allocate(key);
        if (_pairs[bucket].second == INACTIVE) {
            EMH_ENTRY(key, bucket);
            return { {this, bucket}, true };
        } else {
            return { {this, bucket}, false };
        }
    }

    std::pair<iterator, bool> insert(KeyT&& key)
    {
        check_expand_need();
        const auto bucket = find_or_allocate(key);
        if (_pairs[bucket].second == INACTIVE) {
            EMH_ENTRY(std::move(key), bucket);
            return { {this, bucket}, true };
        } else {
            return { {this, bucket}, false };
        }
    }

#if 0
    template <typename Iter>
    inline void insert(Iter begin, Iter end)
    {
        reserve(end - begin + _num_filled);
        for (; begin != end; ++begin) {
            insert(*begin);
        }
    }

    void insert(std::initializer_list<value_type> ilist)
    {
        reserve((size_type)ilist.size() + _num_filled);
        for (auto begin = ilist.begin(); begin != ilist.end(); ++begin) {
            insert(*begin);
        }
    }

    template <typename Iter>
    inline void insert(Iter begin, Iter end)
    {
        Iter citbeg = begin;
        Iter citend = begin;
        reserve(end - begin + _num_filled);
        for (; begin != end; ++begin) {
            if (try_insert_mainbucket(*begin) == INACTIVE) {
                std::swap(*begin, *citend++);
            }
        }

        for (; citbeg != citend; ++citbeg) {
            auto& key = *citbeg;
            const auto bucket = find_or_allocate(key);
            if (_pairs[bucket].second == INACTIVE) {
                EMH_ENTRY(key, bucket);
            }
        }
    }
#endif

    template <typename Iter>
    inline void insert_unique(Iter begin, Iter end)
    {
        reserve(end - begin + _num_filled);
        for (; begin != end; ++begin) {
            insert_unique(*begin);
        }
    }

    /// Same as above, but contains(key) MUST be false
    size_type insert_unique(const KeyT& key)
    {
        check_expand_need();
        auto bucket = find_unique_bucket(key);
        EMH_ENTRY(key, bucket);
        return bucket;
    }

    size_type insert_unique(KeyT&& key)
    {
        check_expand_need();
        auto bucket = find_unique_bucket(key);
        EMH_ENTRY(std::move(key), bucket);
        return bucket;
    }

    //not
    template <class... Args>
    inline std::pair<iterator, bool> emplace(Args&&... args)
    {
        return insert(std::forward<Args>(args)...);
    }

    //no any optimize for position
    template <class... Args>
    iterator emplace_hint(const_iterator position, Args&&... args)
    {
        return insert(std::forward<Args>(args)...);
    }

    std::pair<iterator, bool> try_emplace(const value_type& k)
    {
        return insert(k);
    }

    template <class... Args>
    inline std::pair<iterator, bool> emplace_unique(Args&&... args)
    {
        return insert_unique(std::forward<Args>(args)...);
    }

    // -------------------------------------------------------
    //reset last_bucket collision buc
    //1. bucket <= _last_colls
    //2. bucket <= _mask set _last_colls
    //3. reset
    void clear_bucket(size_type bucket)
    {
        _pairs[bucket].second = INACTIVE;
        _num_filled --;
        if (!std::is_trivially_destructible<KeyT>::value) {
            _pairs[bucket].~PairT();
            _pairs[bucket].second = INACTIVE;
        }

#if EMH_HIGH_LOAD
        if (bucket <= _last_colls)
            return;
        else if (bucket <= _mask) {
            _last_colls = bucket;
            return;
        }
        else if (_last_colls <= _mask)
            _last_colls = _mask;
        if (bucket == ++ _last_colls)
            return;

        const auto last_bucket = _last_colls;
        const auto last_next = _pairs[last_bucket].second;
        auto& key = _pairs[last_bucket].first;
        const auto main_bucket = hash_bucket(key);
        const auto prev_bucket = find_prev_bucket(main_bucket, last_bucket);

        if (!std::is_trivially_destructible<KeyT>::value) {
            new(_pairs + bucket) PairT(std::move(key), last_next != last_bucket ? last_next : bucket);
#ifdef TEST_SLOT_FEATURE
            _pairs[bucket]->slot = bucket;
#endif
            _pairs[last_bucket].~PairT();
        } else {
            _pairs[bucket].first = key;
            _pairs[bucket].second = last_next != last_bucket ? last_next : bucket;
        }

        _pairs[prev_bucket].second = bucket;
        _pairs[last_bucket].second = INACTIVE;
#endif
    }

#ifdef TEST_SLOT_FEATURE
    /// return 0 if not erase
    size_type erase_node(const KeyT& key, const size_type slot)
    {
        assert(slot < _num_buckets && _pairs[slot].second != INACTIVE);
        if (_pairs[slot].first == key) {
            erase_bucket(slot);
            return 1;
        }
        return erase(key);
    }
#endif

    /// Erase an element from the hash table.
    /// return 0 if element was not found
    size_type erase(const KeyT& key)
    {
        const auto bucket = erase_key(key);
        if (bucket == INACTIVE)
            return 0;

        clear_bucket(bucket);
        return 1;
    }

    iterator erase(const_iterator cit)
    {
        iterator it(this, cit._bucket);
        const auto bucket = erase_bucket(it._bucket);
        //move last bucket to current

        //erase from main bucket, return main bucket as next
        return (bucket == it._bucket) ? ++it : it;
    }

    void _erase(const_iterator it)
    {
        erase_bucket(it._bucket);
    }

    void clearkv()
    {
        for (size_type bucket = 0; _num_filled > 0; ++bucket) {
            if (_pairs[bucket].second != INACTIVE) {
                _pairs[bucket].~PairT(); _num_filled --; _pairs[bucket].second = INACTIVE;
            }
        }
    }

    /// Remove all elements, keeping full capacity.
    void clear()
    {
        if (_num_filled > _num_buckets / 4 && std::is_trivially_destructible<KeyT>::value)
            memset(_pairs, INACTIVE, sizeof(_pairs[0]) * _num_buckets);
        else
            clearkv();

        _num_filled = 0;
        _last_colls = _num_buckets - 1;
    }

    void shrink_to_fit() noexcept
    {
        rehash(_num_filled);
    }

    /// Make room for this many elements
    bool reserve(uint64_t num_elems)
    {
        const auto required_buckets = num_elems * _loadlf >> 27;
        if (EMH_LIKELY(required_buckets < _num_buckets))
            return false;
        else if (_num_filled < 16 && _num_filled < _num_buckets)
            return false;

        rehash(required_buckets + 2);
        return true;
    }

    static inline PairT* alloc_bucket(size_type num_buckets)
    {
        auto new_pairs = (char*)malloc((2 + num_buckets) * sizeof(PairT));
        return (PairT*)(new_pairs);
    }

private:

    size_type calculate_bucket(size_type required_buckets)
    {
        size_type num_buckets = 4;
        if (_num_filled > (1 << 16))
            num_buckets = _num_buckets >> 2;
        while (num_buckets < required_buckets)
            num_buckets *= 2;
        return num_buckets;
    }

    void rehash(size_type required_buckets)
    {
        if (required_buckets < _num_filled)
            return;

        auto num_buckets = calculate_bucket(required_buckets);
        if (num_buckets == _num_buckets && _mask != 0)
            return;

        _mask        = num_buckets - 1;
#if EMH_HIGH_LOAD
        num_buckets += num_buckets / 5 + 5;
#endif

        //assert(num_buckets > _num_filled);
        auto new_pairs = (PairT*)alloc_bucket(num_buckets);
        auto old_num_filled  = _num_filled;
        auto old_num_buckets = _num_buckets;
        auto old_pairs = _pairs;

        _num_filled  = 0;
        _num_buckets = num_buckets;
        _pairs       = new_pairs;
        _last_colls  = num_buckets - 1;

        if (bInCacheLine) {
            memset(_pairs, INACTIVE, sizeof(_pairs[0]) * num_buckets);
        } else {
            for (size_type bucket = 0; bucket < num_buckets; bucket++)
                _pairs[bucket].second = INACTIVE;
        }
        memset(_pairs + num_buckets, 0, sizeof(_pairs[0]) * 2);

        //set all main bucket first
        for (size_type src_bucket = 0; _num_filled < old_num_filled; src_bucket++) {
            auto& opairs = old_pairs[src_bucket];
            if (opairs.second == INACTIVE)
                continue;

            const auto bucket = find_unique_bucket(opairs.first);
            EMH_ENTRY(std::move(opairs.first), bucket);
#ifdef TEST_SLOT_FEATURE
            _pairs[bucket].first->slot = bucket;
#endif
            if (!std::is_trivially_destructible<KeyT>::value)
                opairs.~PairT();
        }

#if EMH_REHASH_LOG
        if (_num_filled > EMH_REHASH_LOG) {
            const auto mbucket = bucket_main();
            const auto collision = _num_filled - mbucket;
            char buff[255] = {0};
            sprintf(buff, "    _num_filled/aver_size/type/sizeof/collision/load_factor = %u/%.2lf/%s/%zd/%2.lf%%|%.2f",
            _num_filled, (double)_num_filled / mbucket, typeid(KeyT).name(), sizeof(_pairs[0]), (collision * 100.0 / _num_filled), load_factor());
#ifdef EMH_LOG
            static size_type ihashs = 0;
            EMH_LOG() << "|rhash_nums = " << ihashs ++ << "|" <<__FUNCTION__ << "|" << buff << endl;
#else
            puts(buff);
#endif
        }
#endif

        free(old_pairs);
        assert(old_num_filled == _num_filled);
    }

private:
    // Can we fit another element?
    inline bool check_expand_need()
    {
        return reserve(_num_filled);
    }

    size_type erase_key(const KeyT& key)
    {
        const auto bucket = hash_bucket(key);
        auto next_bucket = _pairs[bucket].second;
        if (next_bucket == INACTIVE)
            return INACTIVE;

        const auto eqkey = _eq(key, _pairs[bucket].first);
        if (next_bucket == bucket) {
            return eqkey ? bucket : INACTIVE;
         } else if (eqkey) {
            const auto nbucket = _pairs[next_bucket].second;
            if (std::is_trivial<KeyT>::value)
                _pairs[bucket].first = _pairs[next_bucket].first;
            else
                std::swap(_pairs[bucket].first, _pairs[next_bucket].first);
#ifdef TEST_SLOT_FEATURE
            _pairs[bucket].first->slot = bucket;
#endif
            _pairs[bucket].second = (nbucket == next_bucket) ? bucket : nbucket;
            return next_bucket;
        }/* else if (EMH_UNLIKELY(bucket != hash_bucket(_pairs[bucket].first)))
            return INACTIVE;
**/
        auto prev_bucket = bucket;
        while (true) {
            const auto nbucket = _pairs[next_bucket].second;
            if (_eq(key, _pairs[next_bucket].first)) {
                _pairs[prev_bucket].second = (nbucket == next_bucket) ? prev_bucket : nbucket;
                return next_bucket;
            }

            if (nbucket == next_bucket)
                break;
            prev_bucket = next_bucket;
            next_bucket = nbucket;
        }

        return INACTIVE;
    }

    //return the real erased bucket
    size_type erase_bucket(const size_type bucket)
    {
        const auto next_bucket = _pairs[bucket].second;
        const auto main_bucket = hash_bucket(_pairs[bucket].first);
        if (bucket == main_bucket) {
            if (bucket != next_bucket) {
                const auto nbucket = _pairs[next_bucket].second;
                if (std::is_trivial<KeyT>::value)
                    _pairs[bucket].first = _pairs[next_bucket].first;
                else
                    std::swap(_pairs[bucket].first, _pairs[next_bucket].first);
                _pairs[bucket].second = (nbucket == next_bucket) ? bucket : nbucket;
#ifdef TEST_SLOT_FEATURE
                _pairs[bucket].first->slot = bucket;
#endif
            }
            clear_bucket(next_bucket);
            return next_bucket;
        }

        const auto prev_bucket = find_prev_bucket(main_bucket, bucket);
        _pairs[prev_bucket].second = (bucket == next_bucket) ? prev_bucket : next_bucket;
        clear_bucket(bucket);
        return bucket;
    }

    // Find the bucket with this key, or return bucket size
    size_type find_filled_bucket(const KeyT& key) const
    {
        const auto bucket = hash_bucket(key);
        auto next_bucket = _pairs[bucket].second;
        const auto& bucket_key = _pairs[bucket].first;
        if (next_bucket == INACTIVE) // || bucket != hash_bucket(bucket_key))
            return _num_buckets;
        else if (_eq(key, bucket_key))
            return bucket;
        else if (next_bucket == bucket)
            return _num_buckets;

        //find next linked bucket
        while (true) {
            if (_eq(key, _pairs[next_bucket].first))
                return next_bucket;

            const auto nbucket = _pairs[next_bucket].second;
            if (nbucket == next_bucket)
                break;
            next_bucket = nbucket;
        }

        return _num_buckets;
    }

    //kick out bucket and find empty to occpuy
    //it will break the orgin link and relnik again.
    //before: main_bucket-->prev_bucket --> bucket   --> next_bucket
    //atfer : main_bucket-->prev_bucket --> (removed)--> new_bucket--> next_bucket
    size_type kickout_bucket(const size_type main_bucket, const size_type bucket)
    {
        const auto next_bucket = _pairs[bucket].second;
        const auto new_bucket  = find_empty_bucket(next_bucket);
        const auto prev_bucket = find_prev_bucket(main_bucket, bucket);
        EMH_MOVE(new_bucket, std::move(_pairs[bucket]));
#ifdef TEST_SLOT_FEATURE
        _pairs[new_bucket].first->slot = new_bucket;
#endif

        _pairs[prev_bucket].second = new_bucket;
        if (next_bucket == bucket)
            _pairs[new_bucket].second = new_bucket;

        if (!std::is_trivially_destructible<KeyT>::value)
            _pairs[bucket].~PairT();

        _pairs[bucket].second = INACTIVE;
        return bucket;
    }

/*****
** inserts a new key into a hash table; first, check whether key's main
** bucket/position is free. If not, check whether colliding node/bucket is in its main
** position or not: if it is not, move colliding bucket to an empty place and
** put new key in its main position; otherwise (colliding bucket is in its main
** position), new key goes to an empty position.
*/
    size_type find_or_allocate(const KeyT& key)
    {
        const auto bucket = hash_bucket(key);
        const auto& bucket_key = _pairs[bucket].first;
        auto next_bucket = _pairs[bucket].second;
        if (next_bucket == INACTIVE || _eq(key, bucket_key))
            return bucket;

        //check current bucket_key is in main bucket or not
        const auto obmain = hash_bucket(bucket_key);
        if (obmain != bucket)
            return kickout_bucket(obmain, bucket);
        else if (next_bucket == bucket)
            return _pairs[next_bucket].second = find_empty_bucket(next_bucket);

        //find next linked bucket and check key
        while (true) {
            if (_eq(key, _pairs[next_bucket].first)) {
#if EMH_LRU_SET
                std::swap(_pairs[bucket].first, _pairs[next_bucket].first);
                return bucket;
#else
                return next_bucket;
#endif
            }

            const auto nbucket = _pairs[next_bucket].second;
            if (nbucket == next_bucket)
                break;
            next_bucket = nbucket;
        }

        //find a new empty and link it to tail
        const auto new_bucket = find_empty_bucket(next_bucket);
        return _pairs[next_bucket].second = new_bucket;
    }

    size_type find_unique_bucket(const KeyT& key)
    {
        const auto bucket = hash_bucket(key);
        auto next_bucket = _pairs[bucket].second;
        if (next_bucket == INACTIVE)
            return bucket;

        //check current bucket_key is in main bucket or not
        const auto obmain = hash_bucket(_pairs[bucket].first);
        if (obmain != bucket)
            return kickout_bucket(obmain, bucket);
        else if (next_bucket != bucket)
            next_bucket = find_last_bucket(next_bucket);

        //find a new empty and link it to tail
        return _pairs[next_bucket].second = find_empty_bucket(next_bucket);
    }


    // key is not in this map. Find a place to put it.
    size_type find_empty_bucket(const size_type bucket_from)
    {
        const auto bucket = bucket_from + 1;
        if (_pairs[bucket].second == INACTIVE)
            return bucket;

        constexpr auto linear_probe_length = std::max((unsigned int)(128 / sizeof(PairT)) + 2, 4u);//cpu cache line 64 byte,2-3 cache line miss
        auto offset = 1u;

        for (; offset < linear_probe_length; offset ++) {
             auto slot = (bucket + offset) & _mask;
             if (_pairs[slot].second == INACTIVE)
                return slot;
        }

        for (size_type step = 1, slot = (offset * 2) & _mask; ; slot = (1 + slot) & _mask, step ++) {
            if (_pairs[slot].second == INACTIVE || _pairs[++slot].second == INACTIVE)
                return slot;

            if (step > 4) {
#if EMH_HIGH_LOAD
                if (INACTIVE == _pairs[_last_colls--].second)
                    return _last_colls + 1;
#else
                if (INACTIVE == _pairs[_last_colls].second || INACTIVE == _pairs[++_last_colls].second)
                    return _last_colls++;
                _last_colls &= _mask;
                //auto tail = (_num_buckets / 2 + _last_colls) & _mask;
                auto tail = _num_buckets - _last_colls;
                if (INACTIVE == _pairs[tail].second || INACTIVE == _pairs[--tail].second)
                    return tail;
#endif
            }
        }
    }

    size_type find_last_bucket(size_type main_bucket) const
    {
        auto next_bucket = _pairs[main_bucket].second;
        if (next_bucket == main_bucket)
            return main_bucket;

        while (true) {
            const auto nbucket = _pairs[next_bucket].second;
            if (nbucket == next_bucket)
                return next_bucket;
            next_bucket = nbucket;
        }
    }

    size_type find_prev_bucket(size_type main_bucket, const size_type bucket) const
    {
        auto next_bucket = _pairs[main_bucket].second;
        if (next_bucket == bucket)
            return main_bucket;

        while (true) {
            const auto nbucket = _pairs[next_bucket].second;
            if (nbucket == bucket)
                return next_bucket;
            next_bucket = nbucket;
        }
    }

    const static uint64_t KC = UINT64_C(11400714819323198485);
    static inline uint64_t hash64(uint64_t key)
    {
#if __SIZEOF_INT128__
        __uint128_t r = key; r *= KC;
        return (uint64_t)(r >> 64) + (uint64_t)r;
#elif _WIN64
        uint64_t high;
        return _umul128(key, KC, &high) + high;
#elif 1
        auto low  =  key;
        auto high = (key >> 32) | (key << 32);
        auto mix  = (0x94d049bb133111ebull * low + 0xbf58476d1ce4e5b9ull * high);
        return mix >> 32;
#elif 1
        uint64_t r = key * UINT64_C(0xca4bcaa75ec3f625);
        return (r >> 32) + r;
#elif 1
        //MurmurHash3Mixer
        uint64_t h = key;
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccd;
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53;
        h ^= h >> 33;
        return h;
#elif 1
        uint64_t x = key;
        x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
        x = x ^ (x >> 31);
        return x;
#endif
    }

    template<typename UType, typename std::enable_if<std::is_integral<UType>::value, size_type>::type = 0>
    inline size_type hash_bucket(const UType key) const
    {
#ifdef EMH_INT_HASH
        return hash64(key) & _mask;
#elif EMH_IDENTITY_HASH
        return (key + (key >> (sizeof(UType) * 4))) & _mask;
#elif EMH_WYHASH64
        return wyhash64(key, KC) & _mask;
#else
        return _hasher(key) & _mask;
#endif
    }

    template<typename UType, typename std::enable_if<std::is_same<UType, std::string>::value, size_type>::type = 0>
    inline size_type hash_bucket(const UType& key) const
    {
#ifdef WYHASH_LITTLE_ENDIAN
        return wyhash(key.data(), key.size(), key.size()) & _mask;
#else
        return _hasher(key) & _mask;
#endif
    }

    template<typename UType, typename std::enable_if<!std::is_integral<UType>::value && !std::is_same<UType, std::string>::value, size_type>::type = 0>
    inline size_type hash_bucket(const UType& key) const
    {
#ifdef EMH_INT_HASH
        return (_hasher(key) * KC) & _mask;
#else
        return _hasher(key) & _mask;
#endif
    }

private:

    //the first cache line packed
    PairT*    _pairs;
    HashT     _hasher;
    EqT       _eq;
    uint32_t   _loadlf;
    size_type  _mask;

    size_type  _num_filled;
    size_type  _last_colls;
    size_type  _num_buckets;
};
} // namespace emhash
#if __cplusplus >= 201103L
//template <class Key, typename Hash = std::hash<Key>> using ktprime_hashset_v8 = emhash8::HashSet<Key, Hash>;
#endif

