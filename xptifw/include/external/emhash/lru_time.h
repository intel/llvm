// By Huang Yuanbing 2019-2022
// bailuzhou@163.com
// version 1.0.0

// LICENSE:
//   This software is dual-licensed to the public domain and under the following
//   license: you are granted a perpetual, irrevocable license to copy, modify,
//   publish, and distribute this file as you see fit.

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
#include <ctime>

// likely/unlikely
#if (__GNUC__ >= 4 || __clang__)
#    define EMHASH_LIKELY(condition) __builtin_expect(condition, 1)
#    define EMHASH_UNLIKELY(condition) __builtin_expect(condition, 0)
#else
#    define EMHASH_LIKELY(condition) condition
#    define EMHASH_UNLIKELY(condition) condition
#endif

#if EMHASH_CACHE_LINE_SIZE < 32
    #define EMHASH_CACHE_LINE_SIZE 64
#endif

#define IS_TIMEOUT(p,b)  (p[b].timeout < nowts())
#define SET_TIMEOUT(b,t) _pairs[b].timeout = nowts() + t

#undef NEW_KVALUE

#define EMH_KEY(p,n)     p[n].first
#define EMH_VAL(p,n)     p[n].second
#define NEXT_BUCKET(p,n) p[n].bucket
#define EMH_PKV(p,n)     p[n]
#define NEW_KVALUE(key, value, bucket) new(_pairs + bucket) PairT(key, value, bucket, _time_out), _num_filled ++

namespace emlru_time {

constexpr uint32_t INACTIVE = 0xFFFFFFFF;

inline static uint32_t nowts()
{
#if EMHASH_LRU_TIME > 0
    return EMHASH_LRU_TIME;
#else
    return time(0);
#endif
}

template <typename First, typename Second>
struct entry {
    entry(const First& key, const Second& value, uint32_t ibucket, uint32_t itimeout = 5)
        :second(value),first(key)
    {
        bucket = ibucket;
        timeout = nowts() + itimeout;
    }

    entry(First&& key, Second&& value, uint32_t ibucket, uint32_t itimeout = 5)
        :second(std::move(value)), first(std::move(key))
    {
        bucket = ibucket;
        timeout = nowts() + itimeout;
    }

    entry(const std::pair<First,Second>& pair, uint32_t itimeout = 5)
        :second(pair.second),first(pair.first)
    {
        bucket = INACTIVE;
        timeout = nowts() + itimeout;
    }

    entry(std::pair<First, Second>&& pair, uint32_t itimeout = 5)
        :second(std::move(pair.second)),first(std::move(pair.first))
    {
        bucket = INACTIVE;
        timeout = nowts() + itimeout;
    }

    entry(const entry& pairT)
        :second(pairT.second),first(pairT.first)
    {
        bucket = pairT.bucket;
        timeout = pairT.timeout;
    }

    entry(entry&& pairT)
        :second(std::move(pairT.second)),first(std::move(pairT.first))
    {
        bucket = pairT.bucket;
        timeout= pairT.timeout;
    }

    entry& operator = (entry&& pairT)
    {
        second = std::move(pairT.second);
        first = std::move(pairT.first);
        bucket = pairT.bucket;
        timeout = pairT.timeout;

        return *this;
    }

    entry& operator = (entry& o)
    {
        second = o.second;
        first  = o.first;
        bucket = o.bucket;
        timeout = o.timeout;
        return *this;
    }

    void swap(entry<First, Second>& o)
    {
        std::swap(second, o.second);
        std::swap(first, o.first);
        std::swap(timeout, o.timeout);
    }

    Second second;//int
    uint32_t bucket;
    uint32_t timeout;
    First first; //long
};// __attribute__ ((packed));

/// A cache-friendly hash table with open addressing, linear/qua probing and power-of-two capacity
template <typename KeyT, typename ValueT, typename HashT = std::hash<KeyT>, typename EqT = std::equal_to<KeyT>>
class lru_cache
{
private:
    typedef lru_cache<KeyT, ValueT, HashT, EqT> htype;
    typedef entry<KeyT, ValueT>             PairT;
    typedef entry<KeyT, ValueT>             value_pair;

public:
    typedef KeyT   key_type;
    typedef ValueT mapped_type;

    typedef  size_t       size_type;
    typedef std::pair<KeyT,ValueT>        value_type;
    typedef  PairT&       reference;
    typedef  const PairT& const_reference;

    class iterator
    {
    public:
        typedef std::forward_iterator_tag iterator_category;
        typedef std::ptrdiff_t            difference_type;
        typedef value_pair                value_type;

        typedef value_pair*               pointer;
        typedef value_pair&               reference;

        iterator() { }
        iterator(htype* hash_map, uint32_t bucket) : _map(hash_map), _bucket(bucket) { }

        iterator& operator++()
        {
            goto_next_element();
            return *this;
        }

        iterator operator++(int)
        {
            auto old_index = _bucket;
            goto_next_element();
            return {_map, old_index};
        }

        reference operator*() const
        {
            return _map->EMH_PKV(_pairs, _bucket);
        }

        pointer operator->() const
        {
            return &(_map->EMH_PKV(_pairs, _bucket));
        }

        bool operator==(const iterator& rhs) const
        {
            return _bucket == rhs._bucket;
        }

        bool operator!=(const iterator& rhs) const
        {
            return _bucket != rhs._bucket;
        }

    private:
        void goto_next_element()
        {
            do {
                _bucket++;
            } while (_map->NEXT_BUCKET(_pairs, _bucket) == INACTIVE);
        }

    public:
        htype* _map;
        uint32_t  _bucket;
    };

    class const_iterator
    {
    public:
        typedef std::forward_iterator_tag iterator_category;
        typedef std::ptrdiff_t            difference_type;
        typedef value_pair                value_type;

        typedef value_pair*               pointer;
        typedef value_pair&               reference;

        const_iterator() { }
        const_iterator(const iterator& proto) : _map(proto._map), _bucket(proto._bucket) { }
        const_iterator(const htype* hash_map, uint32_t bucket) : _map(hash_map), _bucket(bucket) { }

        const_iterator& operator++()
        {
            goto_next_element();
            return *this;
        }

        const_iterator operator++(int)
        {
            auto old_index = _bucket;
            goto_next_element();
            return {_map, old_index};
        }

        reference operator*() const
        {
            return _map->EMH_PKV(_pairs, _bucket);
        }

        pointer operator->() const
        {
            return &(_map->EMH_PKV(_pairs, _bucket));
        }

        bool operator==(const const_iterator& rhs) const
        {
            return _bucket == rhs._bucket;
        }

        bool operator!=(const const_iterator& rhs) const
        {
            return _bucket != rhs._bucket;
        }

    private:
        void goto_next_element()
        {
            do {
                _bucket++;
            } while (_map->NEXT_BUCKET(_pairs, _bucket) == INACTIVE);
        }

    public:
        const htype* _map;
        uint32_t  _bucket;
    };

    // ------------------------------------------------------------------------

    void init()
    {
        _num_buckets = 0;
        _mask = 0;
        _pairs = nullptr;
        _num_filled = 0;
        _time_out = 5;
        _max_buckets = 1 << 30;
        max_load_factor(0.8f);
    }

    lru_cache(uint32_t bucket = 4, uint32_t max_bucket = 1 << 24, int timeout = 3600 * 24 * 365)
    {
        init();
        _time_out = timeout;
        _max_buckets = max_bucket;
        reserve(bucket);
    }

    lru_cache(const lru_cache& other)
    {
        _pairs = (PairT*)malloc((2 + other._num_buckets) * sizeof(PairT));
        clone(other);
    }

    lru_cache(lru_cache&& other)
    {
        init();
        reserve(1);
        *this = std::move(other);
    }

    lru_cache(std::initializer_list<std::pair<KeyT, ValueT>> il)
    {
        init();
        reserve((uint32_t)il.size());
        for (auto begin = il.begin(); begin != il.end(); ++begin)
            insert(*begin);
    }

    lru_cache& operator=(const lru_cache& other)
    {
        if (this == &other)
            return *this;

        if (is_notrivially())
            clearkv();

        if (_num_buckets != other._num_buckets) {
            free(_pairs);
            _pairs = (PairT*)malloc((2 + other._num_buckets) * sizeof(PairT));
        }

        clone(other);
        return *this;
    }

    lru_cache& operator=(lru_cache&& other)
    {
        if (this != &other) {
            swap(other);
            other.clear();
        }
        return *this;
    }

    ~lru_cache()
    {
        if (is_notrivially())
            clearkv();

        free(_pairs);
    }

    void clone(const lru_cache& other)
    {
        _hasher      = other._hasher;
        _num_buckets = other._num_buckets;
        _num_filled  = other._num_filled;
        _mask        = other._mask;
        _loadlf      = other._loadlf;
        _time_out    = other._time_out;
        _max_buckets = other._max_buckets;
        auto opairs  = other._pairs;

        if (std::is_pod<KeyT>::value && std::is_pod<ValueT>::value) {
            memcpy(_pairs, opairs, (_num_buckets + 2) * sizeof(PairT));
        } else {
            for (uint32_t bucket = 0; bucket < _num_buckets; bucket++) {
                auto next_bucket = NEXT_BUCKET(_pairs, bucket) = NEXT_BUCKET(opairs, bucket);
                if (next_bucket != INACTIVE)
                    new(_pairs + bucket) PairT(opairs[bucket]);
            }
            NEXT_BUCKET(_pairs, _num_buckets) = NEXT_BUCKET(_pairs, _num_buckets + 1) = 0;
            _pairs[_num_buckets + 0].timeout = _pairs[_num_buckets + 1].timeout = INACTIVE;
        }
    }

    void swap(lru_cache& other)
    {
        std::swap(_hasher, other._hasher);
        std::swap(_pairs, other._pairs);
        std::swap(_num_buckets, other._num_buckets);
        std::swap(_num_filled, other._num_filled);
        std::swap(_mask, other._mask);
        std::swap(_loadlf, other._loadlf);
        std::swap(_time_out, other._time_out);
        std::swap(_max_buckets, other._max_buckets);
    }

    bool check_timeout(uint32_t bucket)
    {
        //check only main bucket
        if (IS_TIMEOUT(_pairs, bucket) || hash_bucket(EMH_KEY(_pairs, bucket)) == bucket)
        {
            //_pairs[bucket].~PairT();
            clear_bucket(bucket);
            return true;
        }

        return false;
    }

    // -------------------------------------------------------------

    iterator begin()
    {
        uint32_t bucket = 0;
        while (NEXT_BUCKET(_pairs, bucket) == INACTIVE) {
            ++bucket;
        }
        return {this, bucket};
    }

    const_iterator cbegin() const
    {
        uint32_t bucket = 0;
        while (NEXT_BUCKET(_pairs, bucket) == INACTIVE) {
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
        return {this, _num_buckets};
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
        return static_cast<float>(_num_filled) / (_mask + 1);
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
        if (value < 0.95f && value > 0.2f)
            _loadlf = (uint32_t)((1 << 27) / value);
    }

    constexpr size_type max_size() const
    {
        return (1 << 30);
    }

    constexpr size_type max_bucket_count() const
    {
        return (1 << 30);
    }

#ifdef EMHASH_STATIS
    //Returns the bucket number where the element with key k is located.
    size_type bucket(const KeyT& key) const
    {
        const auto bucket = hash_bucket(key);
        const auto next_bucket = NEXT_BUCKET(_pairs, bucket);
        if (next_bucket == INACTIVE)
            return 0;
        else if (bucket == next_bucket)
            return bucket + 1;

        const auto& bucket_key = EMH_KEY(_pairs, bucket);
        return hash_bucket(bucket_key) + 1;
    }

    //Returns the number of elements in bucket n.
    size_type bucket_size(const uint32_t bucket) const
    {
        auto next_bucket = NEXT_BUCKET(_pairs, bucket);
        if (next_bucket == INACTIVE)
            return 0;

        const auto& bucket_key = EMH_KEY(_pairs, bucket);
        next_bucket = hash_bucket(bucket_key);
        uint32_t ibucket_size = 1;

        //iterator each item in current main bucket
        while (true) {
            const auto nbucket = NEXT_BUCKET(_pairs, next_bucket);
            if (nbucket == next_bucket) {
                break;
            }
            ibucket_size ++;
            next_bucket = nbucket;
        }
        return ibucket_size;
    }

    size_type get_main_bucket(const uint32_t bucket) const
    {
        auto next_bucket = NEXT_BUCKET(_pairs, bucket);
        if (next_bucket == INACTIVE)
            return INACTIVE;

        const auto& bucket_key = EMH_KEY(_pairs, bucket);
        const auto main_bucket = hash_bucket(bucket_key);
        return main_bucket;
    }

    size_type get_cache_info(uint32_t bucket, uint32_t next_bucket) const
    {
        auto pbucket = reinterpret_cast<std::uintptr_t>(&_pairs[bucket]);
        auto pnext   = reinterpret_cast<std::uintptr_t>(&_pairs[next_bucket]);
        if (pbucket / 64 == pnext / 64)
            return 0;
        auto diff = pbucket > pnext ? (pbucket - pnext) : pnext - pbucket;
        if (diff < 127 * 64)
            return diff / 64 + 1;
        return 127;
    }

    int get_bucket_info(const uint32_t bucket, uint32_t steps[], const uint32_t slots) const
    {
        auto next_bucket = NEXT_BUCKET(_pairs, bucket);
        if (next_bucket == INACTIVE)
            return -1;

        const auto& bucket_key = EMH_KEY(_pairs, bucket);
        const auto main_bucket = hash_bucket(bucket_key);
        if (main_bucket != bucket)
            return 0;
        else if (next_bucket == bucket)
            return 1;

        steps[get_cache_info(bucket, next_bucket) % slots] ++;
        uint32_t ibucket_size = 2;
        //find a new empty and linked it to tail
        while (true) {
            const auto nbucket = NEXT_BUCKET(_pairs, next_bucket);
            if (nbucket == next_bucket)
                break;

            steps[get_cache_info(nbucket, next_bucket) % slots] ++;
            ibucket_size ++;
            next_bucket = nbucket;
        }
        return ibucket_size;
    }

    void dump_statis() const
    {
        uint32_t buckets[129] = {0};
        uint32_t steps[129]   = {0};
        for (uint32_t bucket = 0; bucket < _num_buckets; ++bucket) {
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
            printf("  %2u  %8u  %.2lf  %.2lf\n", i, bucketsi, bucketsi * 100.0 * i / _num_filled, sumn * 100.0 / _num_filled);
        }

        puts("========== collision miss ration ===========");
        for (uint32_t i = 0; i < sizeof(steps) / sizeof(steps[0]); i++) {
            sumc += steps[i];
            if (steps[i] <= 2)
                continue;
            printf("  %2u  %8u  %.2lf  %.2lf\n", i, steps[i], steps[i] * 100.0 / collision, sumc * 100.0 / collision);
        }

        if (sumb == 0)  return;
        printf("    _num_filled/bucket_size/packed collision/cache_miss/hit_found = %u/%.2lf/%zd/ %.2lf%%/%.2lf%%/%.2lf\n",
                _num_filled, _num_filled * 1.0 / sumb, sizeof(PairT), (collision * 100.0 / _num_filled), (collision - steps[0]) * 100.0 / _num_filled, finds * 1.0 / _num_filled);
        assert(sumn == _num_filled);
        assert(sumc == collision);
    }
#endif

    // ------------------------------------------------------------

    iterator find(const KeyT& key) noexcept
    {
        return {this, find_filled_bucket(key)};
    }

    const_iterator find(const KeyT& key) const noexcept
    {
        return {this, find_filled_bucket(key)};
    }

    bool contains(const KeyT& key) const noexcept
    {
        return find_filled_bucket(key) != _num_buckets;
    }

    size_type count(const KeyT& key) const noexcept
    {
        return find_filled_bucket(key) == _num_buckets ? 0 : 1;
    }

    std::pair<iterator, iterator> equal_range(const KeyT& key)
    {
        const auto found = find(key);
        if (found == end())
            return { found, found };
        else
            return { found, std::next(found) };
    }

    /// Returns the matching ValueT or nullptr if k isn't found.
    bool try_get(const KeyT& key, ValueT& val) const
    {
        const auto bucket = find_filled_bucket(key);
        const auto found = bucket != _num_buckets;
        if (found) {
            val = EMH_VAL(_pairs, bucket);
        }
        return found;
    }

    /// Returns the matching ValueT or nullptr if k isn't found.
    ValueT* try_get(const KeyT& key) noexcept
    {
        const auto bucket = find_filled_bucket(key);
        return bucket == _num_buckets ? nullptr : &EMH_VAL(_pairs, bucket);
    }

    /// Const version of the above
    ValueT* try_get(const KeyT& key) const noexcept
    {
        const auto bucket = find_filled_bucket(key);
        return bucket == _num_buckets ? nullptr : &EMH_VAL(_pairs, bucket);
    }

    /// Convenience function.
    ValueT get_or_return_default(const KeyT& key) const noexcept
    {
        const auto bucket = find_filled_bucket(key);
        return bucket == _num_buckets ? ValueT() : EMH_VAL(_pairs, bucket);
    }

    // -----------------------------------------------------

    /// Returns a pair consisting of an iterator to the inserted element
    /// (or to the element that prevented the insertion)
    /// and a bool denoting whether the insertion took place.
    std::pair<iterator, bool> insert(const KeyT& key, const ValueT& value)
    {
        check_expand_need();
        const auto bucket = find_or_allocate(key);
        auto found = NEXT_BUCKET(_pairs, bucket) == INACTIVE;
        if (found) {
            NEW_KVALUE(key, value, bucket);
        } else {
            if (IS_TIMEOUT(_pairs, bucket)) {
                EMH_KEY(_pairs, bucket) = key;
                EMH_VAL(_pairs, bucket) = value;
                found = true;
            }
            SET_TIMEOUT(bucket, _time_out);
        }
        return { {this, bucket}, found };
    }

    std::pair<iterator, bool> insert(const KeyT& key, const ValueT& value, int timeout) noexcept
    {
        check_expand_need();
        const auto bucket = find_or_allocate(key);
        auto found = NEXT_BUCKET(_pairs, bucket) == INACTIVE;
        if (found) {
            NEW_KVALUE(key, value, bucket);
        } else {
            if (IS_TIMEOUT(_pairs, bucket)) {
                EMH_KEY(_pairs, bucket) = key;
                EMH_VAL(_pairs, bucket) = value;
                found = true;
            }
            SET_TIMEOUT(bucket, timeout);
        }
        return { {this, bucket}, found };
    }

//    std::pair<iterator, bool> insert(const value_pair& value) { return insert(value.first, value.second); }
    std::pair<iterator, bool> insert(KeyT&& key, ValueT&& value) noexcept
    {
        check_expand_need();
        const auto bucket = find_or_allocate(key);
        auto found = NEXT_BUCKET(_pairs, bucket) == INACTIVE;
        if (found) {
            NEW_KVALUE(std::move(key), std::move(value), bucket);
        } else {
            if (IS_TIMEOUT(_pairs, bucket)) {
                EMH_KEY(_pairs, bucket) = std::move(key);
                EMH_VAL(_pairs, bucket) = std::move(value);
                found = true;
            }
            SET_TIMEOUT(bucket, _time_out);
        }
        return { {this, bucket}, found };
    }

    inline std::pair<iterator, bool> insert(const std::pair<KeyT, ValueT>& p)
    {
        return insert(p.first, p.second);
    }

    inline std::pair<iterator, bool> insert(std::pair<KeyT, ValueT>&& p)
    {
        return insert(std::move(p.first), std::move(p.second));
    }

#if 0
    template <typename Iter>
    void insert(Iter begin, Iter end)
    {
        reserve(std::distance(begin, end) + _num_filled);
        for (; begin != end; ++begin) {
            emplace(*begin);
        }
    }

    void insert(std::initializer_list<value_type> ilist)
    {
        reserve(ilist.size() + _num_filled);
        for (auto begin = ilist.begin(); begin != end; ++begin) {
            emplace(*begin);
        }
    }

    template <typename Iter>
    void insert2(Iter begin, Iter end)
    {
        Iter citbeg = begin;
        Iter citend = begin;
        reserve(std::distance(begin, end) + _num_filled);
        for (; begin != end; ++begin) {
            if (try_insert_mainbucket(begin->first, begin->second) == INACTIVE) {
                std::swap(*begin, *citend++);
            }
        }

        for (; citbeg != citend; ++citbeg)
            insert(*citbeg);
    }

    uint32_t try_insert_mainbucket(const KeyT& key, const ValueT& value)
    {
        const auto bucket = hash_bucket(key);
        auto next_bucket = NEXT_BUCKET(_pairs, bucket);
        if (next_bucket != INACTIVE)
            return INACTIVE;

        NEW_KVALUE(key, value, bucket);
        return bucket;
    }
#endif

    template <typename Iter>
    void insert_unique(Iter begin, Iter end)
    {
        reserve(std::distance(begin, end) + _num_filled);
        for (; begin != end; ++begin) {
            insert_unique(*begin);
        }
    }

    /// Same as above, but contains(key) MUST be false
    uint32_t insert_unique(const KeyT& key, const ValueT& value)
    {
        check_expand_need();
        auto bucket = find_unique_bucket(key);
        NEW_KVALUE(key, value, bucket);
        return bucket;
    }

    uint32_t insert_unique(KeyT&& key, ValueT&& value)
    {
        check_expand_need();
        auto bucket = find_unique_bucket(key);
        NEW_KVALUE(std::move(key), std::move(value), bucket);
        return bucket;
    }

    uint32_t insert_unique(entry<KeyT, ValueT>&& pair)
    {
        auto bucket = find_unique_bucket(pair.first);
        NEW_KVALUE(std::move(pair.first), std::move(pair.second), bucket);
        return bucket;
    }

    inline uint32_t insert_unique(std::pair<KeyT, ValueT>&& p)
    {
        return insert_unique(std::move(p.first), std::move(p.second));
    }

    inline uint32_t insert_unique(std::pair<KeyT, ValueT>& p)
    {
        return insert_unique(p.first, p.second);
    }

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

    template<class... Args>
    std::pair<iterator, bool> try_emplace(key_type&& k, Args&&... args)
    {
        return insert(k, std::forward<Args>(args)...).first;
    }

    template <class... Args>
    inline std::pair<iterator, bool> emplace_unique(Args&&... args)
    {
        return insert_unique(std::forward<Args>(args)...);
    }

    std::pair<iterator, bool> insert_or_assign(const KeyT& key, ValueT&& value)
    {
        return insert(key, std::move(value));
    }

    std::pair<iterator, bool> insert_or_assign(KeyT&& key, ValueT&& value)
    {
        return insert(std::move(key), std::move(value));
    }

    /// Like std::map<KeyT,ValueT>::operator[].
    ValueT& operator[](const KeyT& key)
    {
        EMHASH_UNLIKELY(check_expand_need());
        auto bucket = find_or_allocate(key);
        /* Check if inserting a new value rather than overwriting an old entry */
        if (NEXT_BUCKET(_pairs, bucket) == INACTIVE) {
            NEW_KVALUE(key, std::move(ValueT()), bucket);
        } else {
            //TODO:replace the key
            if (IS_TIMEOUT(_pairs, bucket)) {
                EMH_KEY(_pairs, bucket) = key;
                EMH_VAL(_pairs, bucket) = ValueT();
            }

            SET_TIMEOUT(bucket, _time_out);
        }
        return EMH_VAL(_pairs, bucket);
    }

    ValueT& operator[](KeyT&& key)
    {
        EMHASH_UNLIKELY(check_expand_need());
        auto bucket = find_or_allocate(key);
        /* Check if inserting a new value rather than overwriting an old entry */
        if (NEXT_BUCKET(_pairs, bucket) == INACTIVE) {
            NEW_KVALUE(std::move(key), std::move(ValueT()), bucket);
        } else {
            if (IS_TIMEOUT(_pairs, bucket)) {
                EMH_KEY(_pairs, bucket) = std::move(key);
                EMH_VAL(_pairs, bucket) = std::move(ValueT());
            }

            SET_TIMEOUT(bucket, _time_out);
        }
        return EMH_VAL(_pairs, bucket);
    }

    // -------------------------------------------------------
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

    //iterator erase(const_iterator begin_it, const_iterator end_it)
    iterator erase(const_iterator cit)
    {
        iterator it(this, cit._bucket);

        const auto bucket = erase_bucket(it._bucket);
        clear_bucket(bucket);
        //erase from main bucket, return main bucket as next
        return (bucket == it._bucket) ? ++it : it;
    }

    void _erase(const_iterator it)
    {
        const auto bucket = erase_bucket(it._bucket);
        clear_bucket(bucket);
    }

    constexpr bool is_notrivially() noexcept
    {
#if __cplusplus >= 201402L || _MSC_VER > 1600 || __clang__
        return !(std::is_trivially_destructible<KeyT>::value && std::is_trivially_destructible<ValueT>::value);
#else
        return !(std::is_pod<KeyT>::value && std::is_pod<ValueT>::value);
#endif
    }

    void clearkv()
    {
        for (uint32_t bucket = 0; _num_filled > 0; ++bucket) {
            if (NEXT_BUCKET(_pairs, bucket) != INACTIVE)
                clear_bucket(bucket);
        }
    }

    void clear_timeout()
    {
        auto now_ts = nowts();
        for (uint32_t bucket = 0; bucket < _num_buckets; ++bucket) {
            if (NEXT_BUCKET(_pairs, bucket) != INACTIVE && _pairs[bucket].timeout < now_ts) {
                erase_bucket(bucket);
                clear_bucket(bucket);
            }
        }
    }

    /// Remove all elements, keeping full capacity.
    void clear()
    {
        if (is_notrivially() || sizeof(PairT) > EMHASH_CACHE_LINE_SIZE || _num_filled < _num_buckets / 4)
            clearkv();
        else
            memset(_pairs, INACTIVE, sizeof(_pairs[0]) * _num_buckets);

        _num_filled = 0;
    }

    void shrink_to_fit()
    {
        rehash(_num_filled);
    }

    /// Make room for this many elements
    bool reserve(uint32_t num_elems)
    {
        const auto required_buckets = (uint32_t)(((uint64_t)num_elems) * _loadlf >> 27);
        if (EMHASH_LIKELY(required_buckets < _mask))
            return false;

        rehash(required_buckets + 2);
        return true;
    }

    void rehash(uint32_t required_buckets)
    {
        if (required_buckets > 2 * _max_buckets)
            required_buckets = 2 * _max_buckets;

        uint32_t num_buckets = _num_filled > 65536 ? (1 << 16) : 4;
        while (num_buckets < required_buckets) { num_buckets *= 2; }

        auto new_pairs = (PairT*)malloc((2 + num_buckets) * sizeof(PairT));
        auto old_num_filled  = _num_filled;
        auto old_pairs = _pairs;
        _pairs = new_pairs;

        _num_filled  = 0;
        _num_buckets = num_buckets;
        _mask        = num_buckets - 1;
        for (uint32_t bucket = 0; bucket < num_buckets; bucket++) {
            NEXT_BUCKET(_pairs, bucket) = INACTIVE;
            _pairs[bucket].timeout = 0;
        }

        NEXT_BUCKET(_pairs, _num_buckets) = NEXT_BUCKET(_pairs, _num_buckets + 1) = 0;
        _pairs[_num_buckets + 0].timeout = _pairs[_num_buckets + 1].timeout = INACTIVE;

        auto now_ts = nowts();
        for (uint32_t src_bucket = 0; old_num_filled > 0; src_bucket++) {
            if (NEXT_BUCKET(old_pairs, src_bucket) == INACTIVE)
                continue;

            old_num_filled -- ;
            if (old_pairs[src_bucket].timeout > now_ts && _num_filled < _max_buckets) {
                auto& key = EMH_KEY(old_pairs, src_bucket);
                const auto bucket = find_unique_bucket(key);
                NEW_KVALUE(std::move(key), std::move(EMH_VAL(old_pairs, src_bucket)), bucket);
                _pairs[bucket].timeout = old_pairs[src_bucket].timeout;
            }
            old_pairs[src_bucket].~PairT();
        }

#if EMHASH_REHASH_LOG || EMHASH_USER_LOG
        if (_num_filled > 1000000) {
            auto mbucket = _num_filled;
            char buff[255] = {0};
            sprintf(buff, "    _num_filled/aver_size/K.V/pack/ = %u/%2.lf/%s.%s/%zd",
                    _num_filled, double (_num_filled) / mbucket, typeid(KeyT).name(), typeid(ValueT).name(), sizeof(_pairs[0]));
#if EMHASH_USER_LOG
            static uint32_t ihashs = 0;
            FDLOG() << "hash_nums = " << ihashs ++ << "|" <<__FUNCTION__ << "|" << buff << endl;
#else
            puts(buff);
#endif
        }
#endif

        free(old_pairs);
        assert(old_num_filled == 0);
    }

private:
    // Can we fit another element?
    inline bool check_expand_need()
    {
        return reserve(_num_filled);
    }

    void clear_bucket(uint32_t bucket)
    {
        if (is_notrivially())
            _pairs[bucket].~PairT();

        NEXT_BUCKET(_pairs, bucket) = INACTIVE;
        _pairs[bucket].timeout = 0;
        _num_filled --;
    }

    uint32_t erase_key(const KeyT& key)
    {
        const auto bucket = hash_bucket(key);
        auto next_bucket = NEXT_BUCKET(_pairs, bucket);
        if (next_bucket == INACTIVE)
            return INACTIVE;

        const auto eqkey = _eq(key, EMH_KEY(_pairs, bucket));
        if (next_bucket == bucket) {
            return eqkey ? bucket : INACTIVE;
         } else if (eqkey) {
            const auto nbucket = NEXT_BUCKET(_pairs, next_bucket);
            if (is_notrivially())
                EMH_PKV(_pairs, bucket).swap(EMH_PKV(_pairs, next_bucket));
            else
                EMH_PKV(_pairs, bucket) = EMH_PKV(_pairs, next_bucket);

            NEXT_BUCKET(_pairs, bucket) = (nbucket == next_bucket) ? bucket : nbucket;
            return next_bucket;
        }/* else if (EMHASH_UNLIKELY(bucket != hash_bucket(EMH_KEY(_pairs, bucket))))
            return INACTIVE;
        */

        auto prev_bucket = bucket;
        while (true) {
            const auto nbucket = NEXT_BUCKET(_pairs, next_bucket);
            if (_eq(key, EMH_KEY(_pairs, next_bucket))) {
                NEXT_BUCKET(_pairs, prev_bucket) = (nbucket == next_bucket) ? prev_bucket : nbucket;
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
        const auto next_bucket = NEXT_BUCKET(_pairs, bucket);
        const auto main_bucket = hash_bucket(EMH_KEY(_pairs, bucket));
        if (bucket == main_bucket) {
            if (bucket == next_bucket)
                return bucket;

            const auto nbucket = NEXT_BUCKET(_pairs, next_bucket);
            if (is_notrivially())
                EMH_PKV(_pairs, bucket).swap(EMH_PKV(_pairs, next_bucket));
            else {
                const auto timeout = _pairs[bucket].timeout;
                EMH_PKV(_pairs, bucket) = EMH_PKV(_pairs, next_bucket);
                _pairs[next_bucket].timeout = timeout;
            }
            NEXT_BUCKET(_pairs, bucket) = (nbucket == next_bucket) ? bucket : nbucket;
            return next_bucket;
        }

        const auto prev_bucket = find_prev_bucket(main_bucket, bucket);
        NEXT_BUCKET(_pairs, prev_bucket) = (bucket == next_bucket) ? prev_bucket : next_bucket;
        return bucket;
    }

    // Find the bucket with this key, or return bucket size
    uint32_t find_filled_bucket(const KeyT& key) const
    {
        const auto bucket = hash_bucket(key);
        auto next_bucket = NEXT_BUCKET(_pairs, bucket);

        if (next_bucket == INACTIVE)
            return _num_buckets;
        else if (_eq(key, EMH_KEY(_pairs, bucket)))
            return IS_TIMEOUT(_pairs, bucket) ? _num_buckets : bucket;
        else if (next_bucket == bucket)
            return _num_buckets;

        while (true) {
            if (_eq(key, EMH_KEY(_pairs, next_bucket)))
                return (IS_TIMEOUT(_pairs, bucket)) ? _num_buckets : next_bucket;

            const auto nbucket = NEXT_BUCKET(_pairs, next_bucket);
            if (nbucket == next_bucket)
                break;
            next_bucket = nbucket;
        }

        return _num_buckets;
    }

    uint32_t kickout_bucket(const uint32_t main_bucket, const uint32_t bucket)
    {
        const auto next_bucket = NEXT_BUCKET(_pairs, bucket);
        const auto new_bucket  = find_empty_bucket(next_bucket);
        const auto prev_bucket = find_prev_bucket(main_bucket, bucket);
        NEXT_BUCKET(_pairs, prev_bucket) = new_bucket;
        new(_pairs + new_bucket) PairT(std::move(_pairs[bucket])); _num_filled ++;
        if (next_bucket == bucket)
            NEXT_BUCKET(_pairs, new_bucket) = new_bucket;

        clear_bucket(bucket);
        return bucket;
    }

/*
** inserts a new key into a hash table; first, check whether key's main
** bucket/position is free. If not, check whether colliding node/bucket is in its main
** position or not: if it is not, move colliding bucket to an empty place and
** put new key in its main position; otherwise (colliding bucket is in its main
** position), new key goes to an empty position.
*/
    uint32_t find_or_allocate(const KeyT& key)
    {
        const auto bucket = hash_bucket(key);
        const auto& bucket_key = EMH_KEY(_pairs, bucket);
        auto next_bucket = NEXT_BUCKET(_pairs, bucket);
        if (next_bucket == INACTIVE || _eq(key, bucket_key))
            return bucket;

        uint32_t time_bucket = IS_TIMEOUT(_pairs, bucket) ? bucket : INACTIVE;
        //check current bucket_key is in main bucket or not
        const auto main_bucket = hash_bucket(bucket_key);
        if (main_bucket != bucket) {
            return kickout_bucket(main_bucket, bucket);
        } else if (next_bucket == bucket) {
            if (time_bucket != INACTIVE)
                return time_bucket;
            return NEXT_BUCKET(_pairs, next_bucket) = find_empty_bucket(next_bucket);
        }

        //find next linked bucket and check key
        while (true) {
            if (_eq(key, EMH_KEY(_pairs, next_bucket))) {
                return next_bucket;
            } else if (time_bucket == INACTIVE && IS_TIMEOUT(_pairs, next_bucket)) {
                time_bucket = next_bucket;
            }

            const auto nbucket = NEXT_BUCKET(_pairs, next_bucket);
            if (nbucket == next_bucket)
                break;
            next_bucket = nbucket;
        }

        if (time_bucket != INACTIVE)
            return time_bucket;

        //find a new empty and link it to tail
        const auto new_bucket = find_empty_bucket(next_bucket);
        return NEXT_BUCKET(_pairs, next_bucket) = new_bucket;
    }

    // key is not in this map. Find a place to put it.
    uint32_t find_empty_bucket(const uint32_t bucket_from)
    {
        auto bucket = bucket_from + 1;
        if (NEXT_BUCKET(_pairs, bucket) == INACTIVE || NEXT_BUCKET(_pairs, ++bucket) == INACTIVE)
            return bucket;

        //for (uint32_t last = 2, slot = 3; ; slot += last, last = slot - last) {
        for (uint32_t last = 1, slot = 4; ; slot += ++last) {
            auto bucket1 = (bucket_from + slot) & _mask;
            if (NEXT_BUCKET(_pairs, bucket1) == INACTIVE || NEXT_BUCKET(_pairs, ++bucket1) == INACTIVE)
                return bucket1;

            if (last > 4) {
                auto& next = NEXT_BUCKET(_pairs, _num_buckets);
                if (INACTIVE == NEXT_BUCKET(_pairs, next++) || INACTIVE == NEXT_BUCKET(_pairs, next++))
                    return next - 1;

                auto medium = (_num_buckets / 2 + next) & _mask;
                if (INACTIVE == NEXT_BUCKET(_pairs, medium) || INACTIVE == NEXT_BUCKET(_pairs, ++medium))
                    return medium;

                next &= _mask;
            }
        }
    }

    uint32_t find_last_bucket(uint32_t main_bucket) const
    {
        auto next_bucket = NEXT_BUCKET(_pairs, main_bucket);
        if (next_bucket == main_bucket)
            return main_bucket;

        while (true) {
            const auto nbucket = NEXT_BUCKET(_pairs, next_bucket);
            if (nbucket == next_bucket)
                return next_bucket;
            next_bucket = nbucket;
        }
    }

    uint32_t find_prev_bucket(uint32_t main_bucket, const uint32_t bucket) const
    {
        auto next_bucket = NEXT_BUCKET(_pairs, main_bucket);
        if (next_bucket == bucket)
            return main_bucket;

        while (true) {
            const auto nbucket = NEXT_BUCKET(_pairs, next_bucket);
            if (nbucket == bucket)
                return next_bucket;
            next_bucket = nbucket;
        }
    }

    uint32_t find_unique_bucket(const KeyT& key)
    {
        const auto bucket = hash_bucket(key);
        auto next_bucket = NEXT_BUCKET(_pairs, bucket);
        if (next_bucket == INACTIVE)
            return bucket;

        //check current bucket_key is in main bucket or not
        const auto main_bucket = hash_bucket(EMH_KEY(_pairs, bucket));
        if (main_bucket != bucket)
            return kickout_bucket(main_bucket, bucket);
        else if (next_bucket != bucket)
            next_bucket = find_last_bucket(next_bucket);

        //find a new empty and link it to tail
        return NEXT_BUCKET(_pairs, next_bucket) = find_empty_bucket(next_bucket);
    }

    static constexpr uint64_t KC = UINT64_C(11400714819323198485);
    static inline uint64_t hash64(uint64_t key)
    {
#if __SIZEOF_INT128__
        __uint128_t r = key; r *= KC;
        return (uint64_t)(r >> 64) + (uint64_t)r;
#elif _WIN64
        uint64_t high;
        return _umul128(key, KC, &high) + high;
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

    //the first cache line packed
    template<typename UType, typename std::enable_if<std::is_integral<UType>::value, uint32_t>::type = 0>
    inline uint32_t hash_bucket(const UType key) const
    {
#if 1
        return (uint32_t)hash64(key) & _mask;
#elif EMHASH_SAFE_HASH
        if (_hash_inter > 0)
            return (uint32_t)hash64(key) & _mask;

        return _hasher(key);
#elif EMHASH_IDENTITY_HASH
        return (key + (key >> (sizeof(UType) * 4))) & _mask;
#elif WYHASH_LITTLE_ENDIAN0
        return wyhash64(key, _num_buckets) & _mask;
#else
        return _hasher(key) & _mask;
#endif
    }

    template<typename UType, typename std::enable_if<std::is_same<UType, std::string>::value, uint32_t>::type = 0>
    inline uint32_t hash_bucket(const UType& key) const
    {
#ifdef WYHASH_LITTLE_ENDIAN
        return wyhash(key.c_str(), key.size(), key.size()) & _mask;
#elif EMHASH_BKR_HASH
        uint32_t hash = 0;
        for (int i = 0, j = 1; i < key.size(); i += j++)
            hash = key[i] + hash * 131;
        return hash & _mask;
#else
        return _hasher(key) & _mask;
#endif
    }

    template<typename UType, typename std::enable_if<!std::is_integral<UType>::value && !std::is_same<UType, std::string>::value, uint32_t>::type = 0>
    inline uint32_t hash_bucket(const UType& key) const
    {
        return _hasher(key) & _mask;
    }

private:

    PairT*    _pairs;
    HashT     _hasher;
    EqT       _eq;
    uint32_t  _loadlf;
    uint32_t  _num_buckets;
    uint32_t  _mask;

    uint32_t  _max_buckets;
    uint32_t  _num_filled;
    uint32_t  _time_out;
};
} // namespace emhash
#if __cplusplus > 199711
//template <class Key, class Val> using emihash = emhash1::lru_cache<Key, Val, std::hash<Key>>;
#endif
