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

// Scan implementation using the same algorithm described here and used by the CUDPP library.
//
// https://research.nvidia.com/sites/default/files/publications/nvr-2008-003.pdf
//
// For a full overview of various scan implementations see:
//
// https://sites.google.com/site/duanemerrill/ScanTR2.pdf
//
// TODO: There may be some better scan implementations that are described in the second reference. Investigate.

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
        T scan_warp(T* const tile_data, const int idx, const _BinaryOp& op) restrict(amp)
        {
            const int widx = idx & _details::warp_max;

            if (widx >= 1) 
                tile_data[idx] = op(tile_data[idx - 1], tile_data[idx]);
            if ((warp_size > 2) && (widx >= 2))
                tile_data[idx] = op(tile_data[idx - 2], tile_data[idx]);
            if ((warp_size > 4) && (widx >= 4))
                tile_data[idx] = op(tile_data[idx - 4], tile_data[idx]);
            if ((warp_size > 8) && (widx >= 8))
                tile_data[idx] = op(tile_data[idx - 8], tile_data[idx]);
            if ((warp_size > 16) && (widx >= 16))
                tile_data[idx] = op(tile_data[idx - 16], tile_data[idx]);
            if ((warp_size > 32) && (widx >= 32))
                tile_data[idx] = op(tile_data[idx - 32], tile_data[idx]);

            if (_Mode == scan_mode::inclusive)
                return tile_data[idx];
            return (widx > 0) ? tile_data[idx - 1] : T();
        }

        template <int TileSize, scan_mode _Mode, typename _BinaryOp, typename T>
        T scan_tile(T* const tile_data, concurrency::tiled_index<TileSize> tidx, const _BinaryOp& op) restrict(amp)
        {
            static_assert(is_power_of_two<warp_size>::value, "Warp size must be an exact power of 2.");
            const int lidx = tidx.local[0];
            const int warp_id = lidx >> log2<warp_size>::value;

            // Step 1: Intra-warp scan in each warp
            auto val = scan_warp<_Mode, _BinaryOp>(tile_data, lidx, op);
            tidx.barrier.wait_with_tile_static_memory_fence();

            // Step 2: Collect per-warp partial results
            if ((lidx & warp_max) == _details::warp_max)
                tile_data[warp_id] = tile_data[lidx];
            tidx.barrier.wait_with_tile_static_memory_fence();

            // Step 3: Use 1st warp to scan per-warp results
            if (warp_id == 0)
                scan_warp<scan_mode::inclusive>(tile_data, lidx, op);
            tidx.barrier.wait_with_tile_static_memory_fence();

            // Step 4: Accumulate results from Steps 1 and 3
            if (warp_id > 0)
                val = op(tile_data[warp_id - 1], val);
            tidx.barrier.wait_with_tile_static_memory_fence();

            // Step 5: Write and return the final result
            tile_data[lidx] = val;
            tidx.barrier.wait_with_tile_static_memory_fence();
            return val;
        }
    }

    template <int TileSize, scan_mode _Mode, typename _BinaryOp, typename T>
    inline void scan_new(const concurrency::array<T, 1>& input_array, concurrency::array<T, 1>& output_array, const _BinaryOp& op)
    {
        static_assert(TileSize >= _details::warp_size, "Tile size must be at least the size of a single warp.");
        static_assert(TileSize % _details::warp_size == 0, "Tile size must be an exact multiple of warp size.");
        static_assert(TileSize <= (_details::warp_size * _details::warp_size), "Tile size must less than or equal to the square of the warp size.");

        assert(output_array.extent[0] >= _details::warp_size);
        auto compute_domain = output_array.extent.tile<TileSize>().pad();
        concurrency::array<T, 1> tile_results(compute_domain / TileSize);

        // 1 & 2. Scan all tiles and store results in tile_results.
        concurrency::parallel_for_each(compute_domain,
            [=, &input_array, &output_array, &tile_results](concurrency::tiled_index<TileSize> tidx) restrict(amp)
        {
            const int gidx = tidx.global[0];
            const int lidx = tidx.local[0];
            tile_static T tile_data[TileSize];
            tile_data[lidx] = padded_read(input_array, gidx);
            tidx.barrier.wait_with_tile_static_memory_fence();

            auto val = _details::scan_tile<TileSize, _Mode>(tile_data, tidx, amp_algorithms::plus<T>());
            if (lidx == (TileSize - 1))
            {
                tile_results[tidx.tile[0]] = val;
                if (_Mode == scan_mode::exclusive)
                    tile_results[tidx.tile[0]] += input_array[gidx];
            }
            padded_write(output_array, gidx, tile_data[lidx]);
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
            [=, &output_array, &tile_results](concurrency::tiled_index<TileSize> tidx) restrict(amp)
        {
            const int gidx = tidx.global[0];
            if (gidx < output_array.extent[0])
                output_array[gidx] += tile_results[tidx.tile[0]];
        });
    }

    // TODO: Refactor this to remove duplicate code. Also need to decide on final API.

    template <int TileSize, typename InIt, typename OutIt>
    inline void scan_exclusive_new(InIt first, InIt last, OutIt dest_first)
    {
        typedef InIt::value_type T;

        const int size = int(std::distance(first, last));
        concurrency::array<T, 1> in(size);
        concurrency::array<T, 1> out(size);
        concurrency::copy(first, last, in);

        scan_new<TileSize, amp_algorithms::scan_mode::exclusive>(in, out, amp_algorithms::plus<T>());

        concurrency::copy(out, dest_first);
    }

    template <int TileSize, typename InIt, typename OutIt>
    inline void scan_inclusive_new(InIt first, InIt last, OutIt dest_first)
    {
        typedef InIt::value_type T;

        const int size = int(std::distance(first, last));
        concurrency::array<T, 1> in(size);
        concurrency::array<T, 1> out(size);
        concurrency::copy(first, last, in);

        scan_new<TileSize, amp_algorithms::scan_mode::inclusive>(in, out, amp_algorithms::plus<T>());

        concurrency::copy(out, dest_first);
    }
}
