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

        TEST_METHOD(amp_details_radix_sort_key)
        {
            //  0 0000  0  0        8 1000  2  0
            //  1 0001  0  1        9 1001  2  1
            //  2 0010  0  2       10 1010  2  2
            //  3 0011  0  3       11 1011  2  3
            //  4 0100  1  0       12 1100  3  0
            //  5 0101  1  1       13 1101  3  1
            //  6 0110  1  2       14 1110  3  2
            //  7 0111  1  3       15 1111  3  3

            std::array<unsigned, 16> input =    {  3,  2,  1,  6, 10, 11, 13,  0, 15, 10,  5, 14,  4, 12,  9,  8 };
            // Key 0 values, 2 bit key:            3   2   1   2   2   3   1   0   3   2   1   2   0   0   1   0
            // Key 1 values, 2 bit key:            0   0   0   1   2   2   3   0   3   2   1   3   1   3   2   3

            // Sorted key 0 values:               16,  4, 12,  8,  
            // Bin counts:                      4, 4, 5, 3
            // Scan values                      0, 4, 8, 13

            std::array<unsigned, 16> expected = {   };

            array_view<unsigned> input_av(int(input.size()), input);
            amp_algorithms::_details::radix_sort_key<unsigned, 2, 4>(input_av, input_av, 0);
            

            // Assert::IsTrue(are_equal(expected, input_av.section(0, 4)));
        }
    };
}; // namespace amp_algorithms_tests

// TODO: Finish make_array_view, assuming we really need it.

template< typename ConstRandomAccessIterator >
void make_array_view( ConstRandomAccessIterator first, ConstRandomAccessIterator last )
{
    return array_view(std::distance(first, last), first);
}
