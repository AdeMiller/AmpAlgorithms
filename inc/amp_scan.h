/*----------------------------------------------------------------------------
* Copyright (c) Microsoft Corp.
*
* Licensed under the Apache License, Version 2.0 (the "License"); you may not
* use this file except in compliance with the License.  You may obtain a copy
* of the License at http://www.apache.org/licenses/LICENSE-2.0
*
* THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
* WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
* MERCHANTABLITY OR NON-INFRINGEMENT.
*
* See the Apache Version 2.0 License for specific language governing
* permissions and limitations under the License.
*---------------------------------------------------------------------------
*
* C++ AMP standard algorithms library.
*
* This file contains the helpers classes in amp_algorithms::_details namespace
*---------------------------------------------------------------------------*/

#pragma once

#include <amp.h>
#include <assert.h>

namespace amp_algorithms
{
    enum class scan_mode : int
    {
        exclusive = 0,
        inclusive = 1
    };

    namespace _details
    {
        static const int warp_size = 32;
        static const int warp_max = _details::warp_size - 1;

        // TODO: Scan still needs optimizing.

        template <scan_mode _Mode, typename _BinaryOp, typename T>
        T scan_warp(T* const p, const int idx, const _BinaryOp& op) restrict(amp)
        {
            const int widx = idx & _details::warp_max;

            if (widx >= 1) p[idx] = op(p[idx - 1], p[idx]);
            if (widx >= 2 && warp_size > 2) p[idx] = op(p[idx - 2], p[idx]);
            if (widx >= 4 && warp_size > 4) p[idx] = op(p[idx - 4], p[idx]);
            if (widx >= 8 && warp_size > 8) p[idx] = op(p[idx - 8], p[idx]);
            if (widx >= 16 && warp_size > 16) p[idx] = op(p[idx - 16], p[idx]);
            if (widx >= 32 && warp_size > 32) p[idx] = op(p[idx - 32], p[idx]);

            if (_Mode == scan_mode::inclusive)
                return p[idx];
            return (widx > 0) ? p[idx - 1] : T();
        }

        template <int TileSize, scan_mode _Mode, typename _BinaryOp, typename T>
        T scan_tile(T* const p, concurrency::tiled_index<TileSize> tidx, const _BinaryOp& op) restrict(amp)
        {
            static_assert(is_power_of_two<warp_size>::value, "Warp size must be an exact power of 2.");
            const int lidx = tidx.local[0];
            const int warp_id = lidx >> log2<warp_size>::value;

            // Step 1: Intra-warp scan in each warp
            auto val = scan_warp<_Mode, _BinaryOp>(p, lidx, op);
            tidx.barrier.wait_with_tile_static_memory_fence();

            // Step 2: Collect per-warp partial results
            if ((lidx & warp_max) == _details::warp_max)
                p[warp_id] = p[lidx];
            tidx.barrier.wait_with_tile_static_memory_fence();

            // Step 3: Use 1st warp to scan per-warp results
            if (warp_id == 0)
                scan_warp<scan_mode::inclusive>(p, lidx, op);
            tidx.barrier.wait_with_tile_static_memory_fence();

            // Step 4: Accumulate results from Steps 1 and 3
            if (warp_id > 0)
                val = op(p[warp_id - 1], val);
            tidx.barrier.wait_with_tile_static_memory_fence();

            // Step 5: Write and return the final result
            p[lidx] = val;
            tidx.barrier.wait_with_tile_static_memory_fence();
            return val;
        }

        template <int TileSize, scan_mode _Mode, typename _BinaryOp, typename T>
        inline void scan_new(const concurrency::array<T, 1>& in, concurrency::array<T, 1>& out, const _BinaryOp& op)
        {
            static_assert(TileSize >= _details::warp_size, "Tile size must be at least the size of a single warp.");
            static_assert(TileSize % _details::warp_size == 0, "Tile size must be an exact multiple of warp size.");
            static_assert(TileSize <= (_details::warp_size * _details::warp_size), "Tile size must less than or equal to the square of the warp size.");

            const int size = out.extent[0];
            assert(size >= _details::warp_size);
            auto compute_domain = concurrency::extent<1>(size).tile<TileSize>().pad();
            concurrency::array<T, 1> tile_results(compute_domain / TileSize);

            // 1 & 2. Scan all tiles and store results in tile_results.
            concurrency::parallel_for_each(compute_domain,
                [=, &in, &out, &tile_results](concurrency::tiled_index<TileSize> tidx) restrict(amp)
            {
                const int gidx = tidx.global[0];
                const int lidx = tidx.local[0];
                tile_static T tile_data[TileSize];
                tile_data[lidx] = padded_read(in, gidx);
                tidx.barrier.wait_with_tile_static_memory_fence();

                auto val = _details::scan_tile<TileSize, _Mode>(tile_data, tidx, amp_algorithms::plus<T>());
                if (lidx == (TileSize - 1))
                {
                    tile_results[tidx.tile[0]] = val;
                    if (_Mode == scan_mode::exclusive)
                        tile_results[tidx.tile[0]] += in[gidx];
                }
                padded_write(out, gidx, tile_data[lidx]);
            });

            // 3. Scan tile results.
            if (tile_results.extent[0] > TileSize)
            {
                scan_new<TileSize, amp_algorithms::scan_mode::exclusive>(tile_results, tile_results, op);
            }
            else
            {
                concurrency::parallel_for_each(compute_domain,
                    [=, &tile_results](concurrency::tiled_index<TileSize> tidx) restrict(amp)
                {
                    const int gidx = tidx.global[0];
                    const int lidx = tidx.local[0];
                    tile_static T tile_data[TileSize];
                    tile_data[lidx] = tile_results[gidx];
                    tidx.barrier.wait_with_tile_static_memory_fence();

                    _details::scan_tile<TileSize, amp_algorithms::scan_mode::exclusive>(tile_data, tidx, amp_algorithms::plus<T>());

                    tile_results[gidx] = tile_data[lidx];
                    tidx.barrier.wait_with_tile_static_memory_fence();
                });
            }
            // 4. Add the tile results to the individual results for each tile.
            concurrency::parallel_for_each(compute_domain,
                [=, &out, &tile_results](concurrency::tiled_index<TileSize> tidx) restrict(amp)
            {
                const int gidx = tidx.global[0];
                if (gidx < size)
                    out[gidx] += tile_results[tidx.tile[0]];
            });
        }
    }

    // TODO: Refactor this to remove duplicate code. Also need to decide on final API.
    template <int TileSize, typename InIt, typename OutIt>
    inline void scan_exclusive_new(InIt first, InIt last, OutIt dest_first)
    {
        typedef InIt::value_type T;

        const int size = int(distance(first, last));
        concurrency::array<T, 1> in(size);
        concurrency::array<T, 1> out(size);
        copy(first, last, in);

        _details::scan_new<TileSize, amp_algorithms::scan_mode::exclusive>(in, out, amp_algorithms::plus<T>());

        copy(out, dest_first);
    }

    template <int TileSize, typename InIt, typename OutIt>
    inline void scan_inclusive_new(InIt first, InIt last, OutIt dest_first)
    {
        typedef InIt::value_type T;

        const int size = int(distance(first, last));
        concurrency::array<T, 1> in(size);
        concurrency::array<T, 1> out(size);
        copy(first, last, in);

        _details::scan_new<TileSize, amp_algorithms::scan_mode::inclusive>(in, out, amp_algorithms::plus<T>());

        copy(out, dest_first);
    }
}
