/*----------------------------------------------------------------------------
* Copyright © Microsoft Corp.
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
    TEST_CLASS(amp_sort_tests)
    {
        TEST_CLASS_INITIALIZE(initialize_tests)
        {
            set_default_accelerator();
        }

        TEST_METHOD(amp_details_radix_key_value_tests)
        {
            enum parameter
            {
                index = 0,
                value = 1,
                expected = 2
            };
            std::array<std::tuple<int, int, int>, 16> theories =
            { 
                std::make_tuple(0, 3, 3),
                std::make_tuple(0, 1, 1),
                std::make_tuple(1, 3, 0),
                std::make_tuple(1, 12, 3),
                std::make_tuple(1, 13, 0),
            };

            for (auto t : theories)
            {
                Assert::AreEqual(std::get<parameter::expected>(t), radix_key_value<int, 2>(std::get<parameter::value>(t), std::get<parameter::index>(t)));
            }
        }

        TEST_METHOD(amp_details_sort_radix)
        {
            std::array<unsigned, 16> input = {  3,  2,  1,  6, 10, 11, 13, 16, 15, 10,  5, 14,  4, 12,  9,  8 };
            // Key 0 values, 2 bit key:         3   2   1   2   2   3   1   0   3   2   1   2   0   0   1   0

            // Bin counts:                      4, 4, 5, 3
            // Scan values                      0, 4, 8, 13

            //std::array<unsigned, 4> expected = { 3, 2, 1, 6 };
            std::array<int, 4> expected = { 4, 4, 5, 3 };
            //std::array<unsigned, 4> expected = { 0, 4, 9, 12 };
            array_view<unsigned> input_av(int(input.size()), input);
            //amp_algorithms::_details::radix_sort_key<unsigned, 2, 4>(input_av, input_av, 0);
            //radix_sort(input_av);
            
            //histogram_tile<unsigned, 2, 8>(input_av, 0);

            Assert::IsTrue(are_equal(expected, input_av.section(0, 4)));
        }
    };
}; // namespace amp_algorithms_tests

// TODO: Finish make_array_view, assuming we really need it.

template< typename ConstRandomAccessIterator >
void make_array_view( ConstRandomAccessIterator first, ConstRandomAccessIterator last )
{
    return array_view(std::distance(first, last), first);
}
