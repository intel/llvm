// emhash7::HashSet for C++11
// version 1.2.0
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
#include <cstdlib>
#include <type_traits>
#include <cassert>
#include <utility>
#include <cstdint>
#include <functional>
#include <iterator>

#ifdef  EMH_KEY
    #undef  EMH_BUCKET
    #undef  EMH_KEY
    #undef  NEW_KEY
#endif

// likely/unlikely
#if (__GNUC__ >= 4 || __clang__)
#    define EMH_LIKELY(condition)   __builtin_expect(condition, 1)
#    define EMH_UNLIKELY(condition) __builtin_expect(condition, 0)
#else
#    define EMH_LIKELY(condition) condition
#    define EMH_UNLIKELY(condition) condition
#endif

//#define next_coll_bucket(bucket)  ((bucket + 1) & _main_mask + _bucket)
#if 0
    #define hash_main_bucket(key)     (uint32_t)((_hasher(key) & (_mains_buckets - 1)) + _colls_buckets)
    #define next_coll_bucket(bucket)  (bucket) & _main_mask
    #define hash_coll_bucket(key)     (hash_inter(key) & _main_mask)
#elif EMH_HASH
    #define hash_main_bucket(key)     (uint32_t)(_hasher(key) & _main_mask)
    #define hash_coll_bucket(key)     ((hash_inter(key) & _coll_mask) + _mains_buckets)
    #define next_coll_bucket(bucket)  ((bucket) & _coll_mask) + _mains_buckets
#else
    #define hash_main_bucket(key)     (uint32_t)(hash_inter(key) & _main_mask)
    #define hash_coll_bucket(key)     ((_hasher(key) & _coll_mask) + _mains_buckets)
    #define next_coll_bucket(bucket)  ((bucket) & _coll_mask) + _mains_buckets
#endif

#if EMH_CACHE_LINE_SIZE < 32
    #define EMH_CACHE_LINE_SIZE 64
#endif

#define EMH_KEY(p,n)      p[n].first
#define EMH_BUCKET(p,n)   p[n].second

namespace emhash7 {
/// A cache-friendly hash table with open addressing, linear probing and power-of-two capacity
template <typename KeyT, typename HashT = std::hash<KeyT>, typename EqT = std::equal_to<KeyT>>
class HashSet
{
    constexpr static uint32_t INACTIVE = 0xFFFFFFFF;

private:
    typedef  HashSet<KeyT, HashT, EqT> htype;
    typedef  std::pair<KeyT, uint32_t> PairT;

public:
    typedef size_t   size_type;
    typedef KeyT     key_type;
    typedef KeyT     value_type;
    typedef KeyT&    reference;
    typedef const KeyT& const_reference;

    class iterator
    {
    public:
        typedef std::forward_iterator_tag iterator_category;
        typedef size_t                    difference_type;
        typedef KeyT                      value_type;
        typedef value_type*               pointer;
        typedef value_type&               reference;

        iterator() { }
        iterator(htype* hash_set, uint32_t bucket) : _set(hash_set), _bucket(bucket) { }

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
            return _set->EMH_KEY(_pairs, _bucket);
        }

        pointer operator->() const
        {
            return &(_set->EMH_KEY(_pairs, _bucket));
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
            } while (_set->EMH_BUCKET(_pairs, _bucket) == INACTIVE || (_bucket < _set->_mains_buckets && _set->EMH_BUCKET(_pairs, _bucket) % 2 == 0));
        }

    public:
        htype* _set;
        uint32_t  _bucket;
    };

    class const_iterator
    {
    public:
        typedef std::forward_iterator_tag iterator_category;
        typedef size_t                    difference_type;
        typedef const KeyT                value_type;
        typedef value_type*               pointer;
        typedef value_type&               reference;

        const_iterator() { }
        const_iterator(iterator proto) : _set(proto._set), _bucket(proto._bucket) {  }
        const_iterator(const htype* hash_set, uint32_t bucket) : _set(hash_set), _bucket(bucket) {  }

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
            return _set->EMH_KEY(_pairs, _bucket);
        }

        pointer operator->() const
        {
            return &(_set->EMH_KEY(_pairs, _bucket));
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
            } while (_set->EMH_BUCKET(_pairs, _bucket) == INACTIVE || (_bucket < _set->_mains_buckets && _set->EMH_BUCKET(_pairs, _bucket) % 2 == 0));
        }

    public:
        const htype* _set;
        uint32_t  _bucket;
    };

    // ------------------------------------------------------------------------

    void init()
    {
        _colls_buckets = 0;
        _mains_buckets = 0;
        _total_buckets = 0;

        _num_colls = 0;
        _num_mains = 0;
        _main_mask = 0;
        _coll_mask = 0;
        _pairs = nullptr;
        max_load_factor(0.9f);
    }

    HashSet(uint32_t bucket = 4)
    {
        init();
        reserve(bucket);
    }

    HashSet(const HashSet& other)
    {
        _pairs = (PairT*)malloc((1 + other._total_buckets) * sizeof(PairT));
        clone(other);
    }

    void clone(const HashSet& other)
    {
        _hasher        = other._hasher;
        _main_mask     = other._main_mask;
        _coll_mask     = other._coll_mask;

        _colls_buckets = other._colls_buckets;
        _mains_buckets = other._mains_buckets;
        _total_buckets = other._total_buckets;

        _num_colls  = other._num_colls;
        _num_mains     = other._num_mains;

        if (std::is_trivially_copyable<KeyT>::value) {
            memcpy(_pairs, other._pairs, _total_buckets * sizeof(PairT));
        } else {
            auto old_pairs = other._pairs;
            for (uint32_t bucket = 0; bucket < _total_buckets; bucket++) {
                auto next_bucket = EMH_BUCKET(_pairs, bucket) = EMH_BUCKET(old_pairs, bucket);
                if (next_bucket != INACTIVE)
                    new(_pairs + bucket) PairT(old_pairs[bucket]);
            }
        }
        EMH_BUCKET(_pairs, _total_buckets) = 1;
    }

    HashSet(HashSet&& other)
    {
        init();
        reserve(1);
        *this = std::move(other);
    }

    HashSet(std::initializer_list<key_type> il, int n = 8)
    {
        init();
        reserve((uint32_t)il.size());
        for (auto begin = il.begin(); begin != il.end(); ++begin)
            insert(*begin);
    }

    HashSet& operator=(const HashSet& other)
    {
        if (this == &other)
            return *this;

        if (!std::is_trivially_destructible<KeyT>::value)
            clearkv();

        if (_total_buckets != other._total_buckets) {
            free(_pairs);
            _pairs = (PairT*)malloc((1 + other._total_buckets) * sizeof(PairT));
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

    void swap(HashSet& other)
    {
        std::swap(_hasher, other._hasher);
        std::swap(_eq, other._eq);
        std::swap(_loadlf, other._loadlf);
        std::swap(_main_mask, other._main_mask);
        std::swap(_coll_mask, other._coll_mask);

        std::swap(_mains_buckets, other._mains_buckets);
        std::swap(_colls_buckets, other._colls_buckets);
        std::swap(_total_buckets, other._total_buckets);

        std::swap(_num_colls, other._num_colls);
        std::swap(_num_mains, other._num_mains);
        std::swap(_pairs, other._pairs);
    }

    // -------------------------------------------------------------

    iterator begin()
    {
        uint32_t bucket = 0;
        while (EMH_BUCKET(_pairs, bucket) == INACTIVE || (bucket < _mains_buckets && EMH_BUCKET(_pairs, bucket) % 2 == 0)) {
            ++bucket;
        }
        return {this, bucket};
    }

    const_iterator cbegin() const
    {
        uint32_t bucket = 0;
        while (EMH_BUCKET(_pairs, bucket) == INACTIVE || (bucket < _mains_buckets && EMH_BUCKET(_pairs, bucket) % 2 == 0)) {
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
        return {this, _total_buckets};
    }

    const_iterator cend() const
    {
        return {this, _total_buckets};
    }

    const_iterator end() const
    {
        return cend();
    }

    size_type size() const
    {
        return _num_colls + _num_mains;
    }

    bool empty() const
    {
        return size() == 0;
    }

    // Returns the number of buckets.
    size_type bucket_count() const
    {
        return _mains_buckets;
    }

    /// Returns average number of elements per bucket.
    float load_factor() const
    {
        return ((float)size()) / _total_buckets;
        //return (_num_colls / static_cast<float>(_colls_buckets));
        //return (_num_mains / static_cast<float>(_mains_buckets + 1));
    }

    HashT hash_function() const
    {
        return _hasher;
    }

    EqT& key_eq() const
    {
        return _eq;
    }

    constexpr float max_load_factor() const
    {
        return (float)(1 << 13) / _loadlf;
    }

    void max_load_factor(float value)
    {
        if (value < 0.99 && value > 0.2)
            _loadlf = (uint32_t)((1 << 13) / value);
    }

    constexpr size_type max_size() const
    {
        return (1 << 30) / sizeof(PairT);
    }

    constexpr size_type max_bucket_count() const
    {
        return (1 << 30) / sizeof(PairT);
    }

    //Returns the bucket number where the element with key k is located.
    size_type bucket(const KeyT& key) const
    {
        return hash_main_bucket(key);
    }

    //Returns the number of elements in bucket n.
    size_type bucket_size(const size_type bucket) const
    {
        assert(bucket < _mains_buckets);
        return (EMH_BUCKET(_pairs, bucket) + 1) / 2;
    }

#ifdef EMH_STATIS
    size_type get_main_bucket(const uint32_t bucket) const
    {
        auto next_bucket = EMH_BUCKET(_pairs, bucket);
        if (next_bucket == INACTIVE)
            return INACTIVE;

        const auto& bucket_key = EMH_KEY(_pairs, bucket);
        const auto main_bucket = hash_coll_bucket(bucket_key);
        return main_bucket;
    }

    int get_cache_info(uint32_t bucket, uint32_t next_bucket) const
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

    int get_bucket_info(const uint32_t bucket, uint32_t steps[], const uint32_t slots) const
    {
        auto next_bucket = EMH_BUCKET(_pairs, bucket);
        if (next_bucket == INACTIVE)
            return -1;

        const auto& bucket_key = EMH_KEY(_pairs, bucket);
        const auto main_bucket = hash_coll_bucket(bucket_key);
        if (main_bucket != bucket)
            return 0;
        else if (next_bucket == bucket)
            return 1;

        steps[get_cache_info(bucket, next_bucket) % slots] ++;
        uint32_t ibucket_size = 2;
        //find a new empty and linked it to tail
        while (true) {
            const auto nbucket = EMH_BUCKET(_pairs, next_bucket);
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
        uint32_t buckets[129] = {0};
        uint32_t steps[129]   = {0};
        for (uint32_t bucket = 0; bucket < _colls_buckets; ++bucket) {
            auto bsize = get_bucket_info(bucket, steps, 128);
            if (bsize > 0)
                buckets[bsize] ++;
        }

        uint32_t sumb = 0, collision = 0, sumc = 0, finds = 0, sumn = 0;
        puts("============== buckets size ration =========");
        for (uint32_t i = 0; i < sizeof(buckets) / sizeof(buckets[0]); i++) {
            const auto bucketsi = buckets[i];
            if (bucketsi == 0)
                continue;
            sumb += bucketsi;
            sumn += bucketsi * i;
            collision += bucketsi * (i - 1);
            finds += bucketsi * i * (i + 1) / 2;
            printf("  %2u  %8u  %.2lf  %.2lf\n", i, bucketsi, bucketsi * 100.0 * i / _num_colls, sumn * 100.0 / _num_colls);
        }

        puts("========== collision miss ration ===========");
        for (uint32_t i = 0; i < sizeof(steps) / sizeof(steps[0]); i++) {
            sumc += steps[i];
            if (steps[i] <= 2)
                continue;
            printf("  %2u  %8u  %.2lf  %.2lf\n", i, steps[i], steps[i] * 100.0 / collision, sumc * 100.0 / collision);
        }

        if (sumb == 0)  return;
        printf("    _num_colls/bucket_size/packed collision/cache_miss/hit_find = %u/%.2lf/%zd/ %.2lf%%/%.2lf%%/%.2lf\n",
                _num_colls, _num_colls * 1.0 / sumb, sizeof(PairT), (collision * 100.0 / _num_colls), (collision - steps[0]) * 100.0 / _num_colls, finds * 1.0 / _num_colls);
        assert(sumn == _num_colls);
        assert(sumc == collision);
    }
#endif

    // ------------------------------------------------------------

    iterator find(const KeyT& key)
    {
        return {this, find_colls_bucket(key)};
    }

    const_iterator find(const KeyT& key) const
    {
        return {this, find_colls_bucket(key)};
    }

    bool contains(const KeyT& key) const
    {
        return find_colls_bucket(key) != _total_buckets;
    }

    size_type count(const KeyT& key) const
    {
        return find_colls_bucket(key) == _total_buckets ? 0 : 1;
    }

    /// Returns a pair consisting of an iterator to the inserted element
    /// (or to the element that prevented the insertion)
    /// and a bool denoting whether the insertion took place.
    std::pair<iterator, bool> insert(const KeyT& key)
    {
        check_expand_need();

        const auto main_bucket = hash_main_bucket(key);
        auto& bucket_size = EMH_BUCKET(_pairs, main_bucket);

        {
            if (bucket_size == INACTIVE) {
                new_key(key, main_bucket, main_bucket);
                return { {this, main_bucket}, true };
            } else if (_eq(key, EMH_KEY(_pairs, main_bucket)) && bucket_size % 2 > 0) {
                return { {this, main_bucket}, false };
            } else if (bucket_size % 2 == 0) {
                auto next_bucket = find_colls_bucket(key);
                if (next_bucket == _total_buckets) {
                    new_key(key, main_bucket, main_bucket);
                    return { {this, main_bucket}, true };
                }
                else {
                    return { {this, next_bucket}, false };
                }
            }
        }

        const auto bucket = find_or_allocate(key);
        auto next_bucket = EMH_BUCKET(_pairs, bucket);
        if (next_bucket == INACTIVE) {
            new_key(key, bucket, main_bucket);
            return { {this, bucket}, true };
        } else {
            return { {this, bucket}, false };
        }
    }

    void new_key(const KeyT& key, uint32_t bucket, uint32_t main_bucket)
    {
        auto& bucket_size = EMH_BUCKET(_pairs, main_bucket);
        if (bucket < _mains_buckets) {
            bucket_size += 1 + bucket_size % 2;
            new(_pairs + bucket) PairT(key, bucket_size);
            _num_mains += 1;
        } else {
            bucket_size += 2;
            new(_pairs + bucket) PairT(key, bucket);
            _num_colls += 1;
        }
    }

    void del_key(uint32_t bucket, const KeyT& key)
    {
        const auto main_bucket = hash_main_bucket(key);
        auto& bucket_size = EMH_BUCKET(_pairs, main_bucket);
        //assert(bucket_size != INACTIVE);

        bucket_size -= 2;
        _num_colls -= 1;

        if ((int)bucket_size == 0)
            bucket_size = INACTIVE;
        _pairs[bucket].~PairT();
        EMH_BUCKET(_pairs, bucket) = INACTIVE;
    }

    void del_main(uint32_t main_bucket, uint32_t& bucket_size)
    {
        //assert (bucket_size % 2 > 0 && bucket_size != INACTIVE);
        //assert (main_bucket < _mains_buckets);
        bucket_size -= 1;
        _num_mains -= 1;

        if ((int)bucket_size == 0)
            bucket_size = INACTIVE;
        _pairs[main_bucket].~PairT();
    }

#if 0
    std::pair<iterator, bool> insert(KeyT&& key)
    {
        check_expand_need();
        const auto bucket = find_or_allocate(key);
        auto next_bucket = EMH_BUCKET(_pairs, bucket);
        if (next_bucket == INACTIVE || (bucket < _mains_buckets && next_bucket % 2 > 0)) {
            new_key(std::move(key), bucket);
            return { {this, bucket}, true };
        } else {
            return { {this, bucket}, false };
        }
    }

    template <typename Iter>
    inline void insert(Iter begin, Iter end)
    {
        reserve(end - begin + _num_colls);
        for (; begin != end; ++begin) {
            insert(*begin);
        }
    }
#endif

    void insert(std::initializer_list<value_type> ilist)
    {
        reserve((uint32_t)ilist.size() + _num_colls);
        for (auto begin = ilist.begin(); begin != ilist.end(); ++begin) {
            insert(*begin);
        }
    }

    template <typename Iter>
    inline void insert(Iter begin, Iter end)
    {
        Iter citbeg = begin;
        Iter citend = begin;
        reserve(end - begin + _num_colls);
        for (; begin != end; ++begin) {
            if (try_insert_mainbucket(*begin) == INACTIVE) {
                std::swap(*begin, *citend++);
            }
        }

        for (; citbeg != citend; ++citbeg) {
            auto& key = *citbeg;
            const auto bucket = find_or_allocate(key);
            if (EMH_BUCKET(_pairs, bucket) == INACTIVE)
                new_key(key, bucket, hash_main_bucket(key));
        }
    }

    template <typename Iter>
    inline void insert_unique(Iter begin, Iter end)
    {
        reserve(end - begin + _num_colls);
        for (; begin != end; ++begin) {
            insert_unique(*begin);
        }
    }

    /// Same as above, but contains(key) MUST be false
    uint32_t insert_unique(const KeyT& key)
    {
        check_expand_need();
        assert(false);
        auto bucket = find_unique_bucket(key);
        new_key(key, bucket, hash_main_bucket(key));
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
        return insert(std::forward<Args>(args)...).first;
    }
    std::pair<iterator, bool> try_emplace(const key_type& k)
    {
        return insert(k).first;
    }
    template <class... Args>
    inline std::pair<iterator, bool> emplace_unique(Args&&... args)
    {
        return insert_unique(std::forward<Args>(args)...);
    }

    //for private:
    uint32_t try_insert_mainbucket(const KeyT& key)
    {
        const auto main_bucket = hash_main_bucket(key);
        auto& bucket_size = EMH_BUCKET(_pairs, main_bucket);
        if (bucket_size == INACTIVE) {
            new_key(key, main_bucket, main_bucket);
            return main_bucket;
        } else if (_eq(key, EMH_KEY(_pairs, main_bucket))) {
            if (bucket_size % 2 == 0)
                new_key(key, main_bucket, main_bucket);
            return main_bucket;
        } else if (bucket_size % 2 == 0) {
            auto next_bucket = find_colls_bucket(key);
            if (next_bucket == _total_buckets) {
                new_key(key, main_bucket, main_bucket);
                return main_bucket;
            } else {
                return next_bucket;
            }
        }

        const auto bucket = hash_coll_bucket(key);
        if (EMH_BUCKET(_pairs, bucket) == INACTIVE) {
            new_key(key, bucket, main_bucket);
            return bucket;
        }

        return INACTIVE;
    }

    // -------------------------------------------------------

    /// Erase an element from the hash table.
    size_type erase(const KeyT& key)
    {
        const auto main_bucket = hash_main_bucket(key);
        auto& bucket_size = EMH_BUCKET(_pairs, main_bucket);
        if (bucket_size == INACTIVE)
            return 0;

        const auto& bucket_key = EMH_KEY(_pairs, main_bucket);
        if (_eq(key, bucket_key) && bucket_size % 2 > 0) {
            del_main(main_bucket, bucket_size);
            return 1;
        } else if (bucket_size <= 1)
            return 0;

        const auto bucket = erase_key(key);
        if (bucket == INACTIVE)
            return 0;

        del_key(bucket, key);
        return 1;
    }

    iterator erase(const_iterator cit)
    {
        iterator it(this, cit._bucket);
        return erase(it);
    }

    /// Erase an element typedef an iterator.
    /// Returns an iterator to the next element (or end()).
    iterator erase(iterator it)
    {
        if (it._bucket < _mains_buckets) {
            auto& bucket_size = EMH_BUCKET(_pairs, it._bucket);
            del_main(it._bucket, bucket_size);
            return ++it;
        }

        //assert(it->first == EMH_KEY(_pairs, it._bucket));
        const auto bucket = erase_bucket(it._bucket);
        del_key(bucket, EMH_KEY(_pairs, bucket));
        //erase from main bucket, return main bucket as next
        if (bucket == it._bucket)
            ++it;
        return it;
    }

    void clearkv()
    {
        for (uint32_t bucket = _mains_buckets; bucket < _total_buckets && _num_colls > 0; ++bucket) {
            auto& next_bucket = EMH_BUCKET(_pairs, bucket);
            if (next_bucket != INACTIVE) {
                _pairs[bucket].~PairT(); _num_colls -= 1;
                next_bucket = INACTIVE;
            }
        }

        for (uint32_t bucket = 0; bucket < _mains_buckets && _num_mains > 0; ++bucket) {
            auto& next_bucket = EMH_BUCKET(_pairs, bucket);
            if (next_bucket != INACTIVE && next_bucket % 2 > 0) {
                _pairs[bucket].~PairT(); _num_mains -= 1;
                next_bucket = INACTIVE;
            }
        }
        assert(_num_colls == 0 && _num_mains == 0);
    }

    /// Remove all elements, keeping full capacity.
    void clear()
    {
        if (size() > _mains_buckets / 2 && std::is_trivially_destructible<KeyT>::value)
            memset(_pairs, INACTIVE,  sizeof(_pairs[0]) * _total_buckets);
        else
            clearkv();

        _num_colls = _num_mains = 0;
    }

    /// Make room for this many elements
    bool reserve(uint32_t num_elems)
    {
        //auto required_buckets = (uint32_t)(((uint64_t)num_elems * _loadlf) >> 13);
        const auto required_buckets = num_elems * 10 / 8 + 2;
        if (EMH_LIKELY(required_buckets < _colls_buckets))
            return false;

        rehash(required_buckets + 2);
        return true;
    }

    /// Make room for this many elements
    void rehash(uint32_t required_buckets)
    {
        if (required_buckets < _num_colls)
            return ;

        uint32_t num_buckets = _num_colls > 1 << 16 ? 1 << 16 : 8;
        while (num_buckets < required_buckets) { num_buckets *= 2; }

        const auto main_bucket = num_buckets;
        auto new_pairs = (PairT*)malloc((1 + num_buckets + main_bucket) * sizeof(PairT));
        auto old_pairs = _pairs;

        const auto old_num_mains   = _num_mains;
        const auto old_num_colls   = _num_colls;
        const auto old_main_buckets  = _mains_buckets;
        const auto old_total_buckets = _total_buckets;
        const auto old_colls_buckets = _colls_buckets;

        _colls_buckets  = num_buckets;
        _mains_buckets  = main_bucket;
        _total_buckets  = _colls_buckets + _mains_buckets;

        _main_mask      = _mains_buckets - 1;
        _coll_mask      = _colls_buckets - 1;
        _pairs       = new_pairs;
        _num_mains   = 0;
        _num_colls   = 0;

        if (sizeof(PairT) <= EMH_CACHE_LINE_SIZE / 2)
            memset(_pairs, INACTIVE, _total_buckets * sizeof(_pairs[0]));
        else {
            for (uint32_t bucket = 0; bucket < _total_buckets; bucket++)
                EMH_BUCKET(_pairs, bucket) = INACTIVE;
        }
        EMH_BUCKET(_pairs, _total_buckets) = 1;

        uint32_t collision = 0;
        //set all main bucket first
        for (uint32_t src_bucket = 0; src_bucket < old_total_buckets; src_bucket++) {
            auto bucket_size = EMH_BUCKET(old_pairs, src_bucket);
            if (bucket_size == INACTIVE || (bucket_size % 2 == 0 && bucket_size < old_main_buckets))
                continue;

            auto& old_pair = old_pairs[src_bucket];
            auto& key = EMH_KEY(old_pairs, src_bucket);

#if 0
            auto bucket = try_insert_mainbucket(key);
            if (bucket == INACTIVE) {
                EMH_BUCKET(old_pairs, collision++) = src_bucket;
            } else {
                old_pair.~PairT();
            }
#else
            const auto main_bucket = hash_main_bucket(key);
            auto& next_bucket = EMH_BUCKET(_pairs, main_bucket);
            next_bucket += 2;

            if (next_bucket == 1) {
                new(_pairs + main_bucket) PairT(std::move(key), next_bucket); old_pair.~PairT();
                _num_mains ++;
            } else {
                const auto bucket = hash_coll_bucket(key);
                auto& next_bucket2 = EMH_BUCKET(_pairs, bucket);
                if (next_bucket2 == INACTIVE) {
                    new(_pairs + bucket) PairT(std::move(old_pair)); old_pair.~PairT();
                    next_bucket2 = bucket;
                    _num_colls ++;
                } else {
                    //move collision bucket to head for better cache performance
                    EMH_BUCKET(old_pairs, collision++) = src_bucket;
                }
            }
#endif
        }

        _num_colls += collision;
        //reset all collisions bucket
        for (uint32_t colls = 0; colls < collision; colls++) {
            const auto src_bucket = EMH_BUCKET(old_pairs, colls);
            const auto main_bucket = hash_coll_bucket(EMH_KEY(old_pairs, src_bucket));
            auto& old_pair = old_pairs[src_bucket];

            auto next_bucket = EMH_BUCKET(_pairs, main_bucket);
            //assert(next_bucket != INACTIVE);
            //check current bucket_key is in main bucket or not
            if (next_bucket != main_bucket)
                next_bucket = find_last_bucket(next_bucket);
            //find a new empty and link it to tail
            auto new_bucket = EMH_BUCKET(_pairs, next_bucket) = find_empty_bucket(next_bucket);
            new(_pairs + new_bucket) PairT(std::move(old_pair)); old_pair.~PairT();
            EMH_BUCKET(_pairs, new_bucket) = new_bucket;
        }

#if EMH_REHASH_LOG
        if (_num_colls > 100000) {
            auto mbucket = size() - collision;
            char buff[255] = {0};
            sprintf(buff, "    _num_colls/main_factor/coll_factor/K/pack/collision = %u/%.2lf%%/%.2lf%%/%s/%zd/%.2lf%%",
                    _num_colls, old_num_mains * 100.0 / old_main_buckets, 100.0 * _num_colls / _colls_buckets, typeid(KeyT).name(), sizeof(_pairs[0]), _num_colls * 100.0 / size());
#ifdef EMH_LOG
            static uint32_t ihashs = 0;
            EMH_LOG() << "|rhash_nums = " << ihashs ++ << "|" <<__FUNCTION__ << "|" << buff << endl;
#else
            puts(buff);
#endif
        }
#endif

        free(old_pairs);

        auto diff = old_num_colls + old_num_mains - _num_colls - _num_mains;
        if (diff != 0) {
            printf("%d %d | %d %d diff = %d\n", old_num_colls, old_num_mains, _num_colls, _num_mains, (int)diff);
            assert(diff == 0);
        }
    }

private:
    // Can we fit another element?
    inline bool check_expand_need()
    {
        return reserve(_num_colls);
    }

    uint32_t erase_key(const KeyT& key)
    {
        const auto bucket = hash_coll_bucket(key);
        auto next_bucket = EMH_BUCKET(_pairs, bucket);
        if (next_bucket == INACTIVE)
            return INACTIVE;

        const auto eqkey = _eq(key, EMH_KEY(_pairs, bucket));
        if (next_bucket == bucket)
            return eqkey ? bucket : INACTIVE;
        else if (eqkey) {
            const auto nbucket = EMH_BUCKET(_pairs, next_bucket);
            if (std::is_trivial<KeyT>::value)
                std::swap(EMH_KEY(_pairs, bucket), EMH_KEY(_pairs, next_bucket));
            else
                EMH_KEY(_pairs, bucket) = EMH_KEY(_pairs, next_bucket);
            EMH_BUCKET(_pairs, bucket) = (nbucket == next_bucket) ? bucket : nbucket;
            return next_bucket;
        } else if (EMH_UNLIKELY(bucket != hash_coll_bucket(EMH_KEY(_pairs, bucket))))
            return INACTIVE;

        auto prev_bucket = bucket;
        while (true) {
            const auto nbucket = EMH_BUCKET(_pairs, next_bucket);
            if (_eq(key, EMH_KEY(_pairs, next_bucket))) {
                EMH_BUCKET(_pairs, prev_bucket) = (nbucket == next_bucket) ? prev_bucket : nbucket;
                return next_bucket;
            }

            if (nbucket == next_bucket)
                break;
            prev_bucket = next_bucket;
            next_bucket = nbucket;
        }

        return INACTIVE;
    }

    uint32_t erase_bucket(const uint32_t bucket)
    {
        const auto next_bucket = EMH_BUCKET(_pairs, bucket);
        const auto main_bucket = hash_coll_bucket(EMH_KEY(_pairs, bucket));
        if (bucket == main_bucket) {
            if (bucket != next_bucket) {
                const auto nbucket = EMH_BUCKET(_pairs, next_bucket);
                if (std::is_trivial<KeyT>::value)
                    std::swap(EMH_KEY(_pairs, bucket), EMH_KEY(_pairs, next_bucket));
                else
                    EMH_KEY(_pairs, bucket) = EMH_KEY(_pairs, next_bucket);
                EMH_BUCKET(_pairs, bucket) = (nbucket == next_bucket) ? bucket : nbucket;
            }
            return next_bucket;
        }

        const auto prev_bucket = find_prev_bucket(main_bucket, bucket);
        EMH_BUCKET(_pairs, prev_bucket) = (bucket == next_bucket) ? prev_bucket : next_bucket;
        return bucket;
    }

    // Find the bucket with this key, or return bucket size
    uint32_t find_colls_bucket(const KeyT& key) const
    {
        const auto main_bucket = hash_main_bucket(key);
        const auto bucket_size = EMH_BUCKET(_pairs, main_bucket);

        {
            if (bucket_size == INACTIVE)
                return _total_buckets;
            else if (_eq(key, EMH_KEY(_pairs, main_bucket)) && bucket_size % 2 > 0)
                return main_bucket;
            else if (bucket_size == 1)
                return _total_buckets;
        }

        const auto bucket = hash_coll_bucket(key);
        auto next_bucket = EMH_BUCKET(_pairs, bucket);
        if (next_bucket == INACTIVE)
            return _total_buckets;
        else if (_eq(key, EMH_KEY(_pairs, bucket)))
            return bucket;
        else if (next_bucket == bucket)
            return _total_buckets;

        //find next linked bucket
        while (true) {
            if (_eq(key, EMH_KEY(_pairs, next_bucket)))
                return next_bucket;

            const auto nbucket = EMH_BUCKET(_pairs, next_bucket);
            if (nbucket == next_bucket)
                break;
            next_bucket = nbucket;
        }

        return _total_buckets;
    }

    uint32_t reset_coll_bucket(const uint32_t main_bucket, const uint32_t bucket)
    {
        const auto next_bucket = EMH_BUCKET(_pairs, bucket);
        const auto new_bucket  = find_empty_bucket(next_bucket);
        const auto prev_bucket = find_prev_bucket(main_bucket, bucket);
        EMH_BUCKET(_pairs, prev_bucket) = new_bucket;
        new(_pairs + new_bucket) PairT(std::move(_pairs[bucket])); _pairs[bucket].~PairT();
        EMH_BUCKET(_pairs, new_bucket) = (next_bucket == bucket) ? new_bucket : next_bucket;
        EMH_BUCKET(_pairs, bucket) = INACTIVE;
        return bucket;
    }

/*
** inserts a new key into a hash table; first check whether key's main
** bucket/position is free. If not, check whether colliding node/bucket is in its main
** position or not: if it is not, move colliding bucket to an empty place and
** put new key in its main position; otherwise (colliding bucket is in its main
** position), new key goes to an empty position.
*/
    uint32_t find_or_allocate(const KeyT& key)
    {
        const auto bucket = hash_coll_bucket(key);
        const auto& bucket_key = EMH_KEY(_pairs, bucket);
        auto next_bucket = EMH_BUCKET(_pairs, bucket);
        if (next_bucket == INACTIVE || _eq(key, bucket_key))
            return bucket;

        //check current bucket_key is in main bucket or not
        const auto main_bucket = hash_coll_bucket(bucket_key);
        if (main_bucket != bucket)
            return reset_coll_bucket(main_bucket, bucket);
        else if (next_bucket == bucket)
            return EMH_BUCKET(_pairs, next_bucket) = find_empty_bucket(next_bucket);

        //find next linked bucket and check key
        while (true) {
            if (_eq(key, EMH_KEY(_pairs, next_bucket))) {
#if EMH_LRU_SET
                std::swap(EMH_KEY(_pairs, bucket), EMH_KEY(_pairs, next_bucket));
                return bucket;
#else
                return next_bucket;
#endif
            }

            const auto nbucket = EMH_BUCKET(_pairs, next_bucket);
            if (nbucket == next_bucket)
                break;
            next_bucket = nbucket;
        }

        //find a new empty and link it to tail
        const auto new_bucket = find_empty_bucket(next_bucket);
        return EMH_BUCKET(_pairs, next_bucket) = new_bucket;
    }

    // key is not in this map. Find a place to put it.
    uint32_t find_empty_bucket(uint32_t bucket_from) const
    {
        const auto bucket = ++bucket_from;
        if (EMH_BUCKET(_pairs, bucket) == INACTIVE)
            return bucket;

        bucket_from = next_coll_bucket(bucket_from + 1);
        if (EMH_BUCKET(_pairs, bucket_from) == INACTIVE)
            return bucket_from;

        constexpr auto max_probe_length = 2 + EMH_CACHE_LINE_SIZE / sizeof(PairT);//cpu cache line 64 byte,2-3 cache line miss

#if 0
        for (auto slot = 1; ; ++slot) {
            const auto bucket1 = next_coll_bucket(bucket_from + slot);
            if (EMH_BUCKET(_pairs, bucket1) == INACTIVE)
                return bucket1;

            const auto bucket2 = next_coll_bucket(bucket1 + 1);
            if (EMH_BUCKET(_pairs, bucket2) == INACTIVE)
                return bucket2;

            bucket_from += slot;
            if (slot > 6) {
                bucket_from += _colls_buckets / 2;
                slot = 1;
            }
        }
#else
        for (uint32_t slot = 1, step = 2; ; slot += ++step) {
            auto bucket = next_coll_bucket(bucket_from + slot);
            if (EMH_BUCKET(_pairs, bucket) == INACTIVE || EMH_BUCKET(_pairs, ++bucket) == INACTIVE)
                return bucket;

            if (step >= max_probe_length) {
                auto bucket3 = next_coll_bucket(_num_mains + step);
                if (EMH_BUCKET(_pairs, bucket3) == INACTIVE || EMH_BUCKET(_pairs, ++bucket3) == INACTIVE)
                    return bucket3;

                auto bucket2 = next_coll_bucket(bucket + slot * slot); //switch to square search
                if (EMH_BUCKET(_pairs, bucket2) == INACTIVE || EMH_BUCKET(_pairs, ++bucket2) == INACTIVE)
                    return bucket2;
            }
        }
#endif
    }

    uint32_t find_last_bucket(uint32_t main_bucket) const
    {
        auto next_bucket = EMH_BUCKET(_pairs, main_bucket);
        if (next_bucket == main_bucket)
            return main_bucket;

        while (true) {
            const auto nbucket = EMH_BUCKET(_pairs, next_bucket);
            if (nbucket == next_bucket)
                return next_bucket;
            next_bucket = nbucket;
        }
    }

    uint32_t find_prev_bucket(uint32_t main_bucket, const uint32_t bucket) const
    {
        auto next_bucket = EMH_BUCKET(_pairs, main_bucket);
        if (next_bucket == bucket)
            return main_bucket;

        while (true) {
            const auto nbucket = EMH_BUCKET(_pairs, next_bucket);
            if (nbucket == bucket)
                return next_bucket;
            next_bucket = nbucket;
        }
    }

    uint32_t find_unique_bucket(const KeyT& key)
    {
        assert(false);
        const auto bucket = hash_coll_bucket(key);
        auto next_bucket = EMH_BUCKET(_pairs, bucket);
        if (next_bucket == INACTIVE)
            return bucket;

        //check current bucket_key is in main bucket or not
        const auto main_bucket = hash_coll_bucket(EMH_KEY(_pairs, bucket));
        if (main_bucket != bucket)
            return reset_coll_bucket(main_bucket, bucket);
        else if (next_bucket != bucket)
            next_bucket = find_last_bucket(next_bucket);

        //find a new empty and link it to tail
        return EMH_BUCKET(_pairs, next_bucket) = find_empty_bucket(next_bucket);
    }

    static inline uint64_t hash64(uint64_t key)
    {
#if __SIZEOF_INT128__
        constexpr uint64_t k = UINT64_C(11400714819323198485);
        __uint128_t r = key; r *= k;
        return (uint32_t)(r >> 64) + (uint32_t)r;
#elif _WIN64
        uint64_t high;
        constexpr uint64_t k = UINT64_C(11400714819323198485);
        return _umul128(key, k, &high) + high;
#elif 1
        uint64_t const r = key * UINT64_C(0xca4bcaa75ec3f625);
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

    template<typename UType, typename std::enable_if<std::is_integral<UType>::value, uint32_t>::type = 0>
    inline uint32_t hash_inter(const UType key) const
    {
#ifndef EMH_INT_HASH
        return hash64(key);
#elif EMH_IDENTITY_HASH
        return key + (key >> (sizeof(UType) * 4));
#else
        return _hasher(key);
#endif
    }

    template<typename UType, typename std::enable_if<!std::is_integral<UType>::value, uint32_t>::type = 0>
    inline uint32_t hash_inter(const UType& key) const
    {
#ifndef EMH_INT_HASH
        return (_hasher(key) * 11400714819323198485ull);
#else
        return _hasher(key);
#endif
    }

private:

    //the first cache line packed
    HashT     _hasher;
    EqT       _eq;
    uint32_t  _loadlf;
    uint32_t  _main_mask;
    uint32_t  _coll_mask;

    uint32_t  _colls_buckets;
    uint32_t  _mains_buckets;
    uint32_t  _total_buckets;

    uint32_t  _num_colls;
    uint32_t  _num_mains;
    PairT*    _pairs;
};
} // namespace emhash
#if __cplusplus > 199711
//template <class Key> using emihash = emhash1::HashSet<Key, std::hash<Key>>;
#endif
