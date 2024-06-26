/*----------------------------------------------------------------------------
* Copyright � Microsoft Corp.
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
* C++ AMP standard algorithm library.
*
* This file contains the unit tests for sort.
*---------------------------------------------------------------------------*/
#include "stdafx.h"

#include <amp_algorithms.h>
#include "testtools.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace concurrency;
using namespace amp_algorithms;
using namespace amp_algorithms::_details;
using namespace testtools;

namespace amp_algorithms_tests
{
    TEST_CLASS_CATEGORY(amp_sort_tests, "amp")
    // {
        TEST_CLASS_INITIALIZE(initialize_tests)
        {
            set_default_accelerator(L"amp_sort_tests");
        }

        TEST_METHOD(amp_details_radix_key_value_tests)
        {
            enum parameter
            {
                index = 0,
                value = 1,
                expected = 2
            };

            std::array<std::tuple<unsigned, int, int>, 5> theories =
            {
                std::make_tuple(0, 3, 3),   // 000010 => ----10
                std::make_tuple(0, 1, 1),   // 000001 => ----01
                std::make_tuple(1, 3, 0),   // 000011 => --00--
                std::make_tuple(1, 13, 3),  // 001101 => --11--
                std::make_tuple(2, 45, 2),  // 101101 => 10----
            };
            
            for (auto t : theories)
            {
                int result = radix_key_value<int, 2>(std::get<parameter::value>(t), std::get<parameter::index>(t));
                Assert::AreEqual(std::get<parameter::expected>(t), result);
            }
        }

        TEST_METHOD(amp_details_radix_sort_tile_by_key_0)
        {
            //  0 0000  0  0        8 1000  2  0
            //  1 0001  0  1        9 1001  2  1
            //  2 0010  0  2       10 1010  2  2
            //  3 0011  0  3       11 1011  2  3
            //  4 0100  1  0       12 1100  3  0
            //  5 0101  1  1       13 1101  3  1
            //  6 0110  1  2       14 1110  3  2
            //  7 0111  1  3       15 1111  3  3

            std::array<unsigned, 16> input =     { 3,  2,  1,  6, 10, 11, 13,  0, 15, 10,  5, 14,  4, 12,  9,  8 };
            // Key 0 values, 2 bit key:            3   2   1   2   2   3   1   0   3   2   1   2   0   0   1   0
            std::array<unsigned, 16> expected  = { 1,  2,  6,  3,  0, 13, 10, 11,  5, 10, 14, 15,  4, 12,  8,  9 };

            array_view<unsigned> input_av(int(input.size()), input);

            concurrency::tiled_extent<4> compute_domain = input_av.get_extent().tile<4>().pad();

            concurrency::parallel_for_each(compute_domain,
                [=](concurrency::tiled_index<4> tidx) restrict(amp)
            {
                const int gidx = tidx.global[0];
                const int idx = tidx.local[0];
                tile_static int tile_data[4];

                tile_data[idx] = input_av[gidx];

                amp_algorithms::_details::radix_sort_tile_by_key<int, 2, 4>(tile_data, tidx, 0);

                input_av[gidx] = tile_data[idx];
            });
            Assert::IsTrue(are_equal(expected, input_av));
        }

        TEST_METHOD(amp_details_radix_sort_tile_by_key_1)
        {
            std::array<unsigned, 16> input  =    { 1,  2,  6,  3,  0, 13, 10, 11,  5, 10, 14, 15,  4, 12,  8,  9 };
            // Key 1 values, 2 bit key:            0   0   1   0   0   3   2   2   1   2   3   3   1   3   2   2
            std::array<unsigned, 16> expected  = { 1,  2,  3,  6,  0, 10, 11, 13,  5, 10, 14, 15,  4,  8,  9, 12 };

            array_view<unsigned> input_av(int(input.size()), input);

            concurrency::tiled_extent<4> compute_domain = input_av.get_extent().tile<4>().pad();

            concurrency::parallel_for_each(compute_domain,
                [=](concurrency::tiled_index<4> tidx) restrict(amp)
            {
                const int gidx = tidx.global[0];
                const int idx = tidx.local[0];
                tile_static int tile_data[4];
                tile_data[idx] = input_av[gidx];

                amp_algorithms::_details::radix_sort_tile_by_key<int, 2, 4>(tile_data, tidx, 1);

                input_av[gidx] = tile_data[idx];
            });
            Assert::IsTrue(are_equal(expected, input_av));
        }

#ifdef _DEBUG
        TEST_METHOD(amp_details_radix_sort_by_key)
        {
            std::array<unsigned, 16> input  =    { 1,  2,  6,  3,  0, 13, 10, 11,  5, 10, 14, 15,  4, 12,  8,  9 };
            std::array<unsigned, 16> expected  = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 };
            std::array<unsigned, 16> output;
            array_view<unsigned> input_av(int(input.size()), input);
            array_view<unsigned> output_av(int(output.size()), output);
            radix_sort_by_key<unsigned, 2, 4>(input_av, output_av, 0);

            output_av.synchronize();

            Assert::IsTrue(are_equal(expected, output_av));
        }
#endif
    };
}; // namespace amp_algorithms_tests

// TODO: Finish make_array_view, assuming we really need it.

template< typename ConstRandomAccessIterator >
void make_array_view( ConstRandomAccessIterator first, ConstRandomAccessIterator last )
{
    return array_view(std::distance(first, last), first);
}
