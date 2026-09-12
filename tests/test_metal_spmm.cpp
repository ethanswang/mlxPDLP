// Copyright 2026 Ethan Wang <ethanshurui.wang@gmail.com>
// SPDX-License-Identifier: Apache-2.0
#include "metal_spmm.h"
#include "metal_spmv.h"
#include <iostream>
#include <random>
#include <stdexcept>
using namespace mlxpdlp;
int main() {
    try {
        if (!mx::is_available(mx::Device::gpu)) return 77;
        auto stream = mx::default_stream(mx::Device::gpu);
        mx::StreamContext context(stream);
        std::mt19937 rng(731);
        for (int columns : {0,33,65536,65537}) for (int tile : {4,8})
        for (int width : {1,2,3,4,5,8,9,16,17}) {
            int padded = (width+tile-1)/tile*tile;
            constexpr int rows = 7;
            std::vector<int32_t> rp{0}, ci;
            std::vector<float> values;
            for (int r = 0; r < rows; ++r) {
                int length = columns == 0 ? 0 : (r == 0 ? 0 : (r == 6 ? 5000 : r*7));
                for (int k = 0; k < length; ++k) {
                    ci.push_back(k % 5 == 0 ? columns-1 : int(rng() % columns));
                    values.push_back(float(int(rng()%19)-9)/16);
                }
                rp.push_back(int(ci.size()));
            }
            std::vector<float> x(size_t(columns)*padded, NAN);
            for (int i = 0; i < columns; ++i) for (int j = 0; j < width; ++j)
                x[size_t(i)*padded+j] = float(int(rng()%23)-11)/8;
            std::vector<int> active(padded, 0); std::fill_n(active.begin(), width, 1);
            auto starts = mx::array(rp.data(), {rows+1}, mx::int32);
            auto indices = detail::metal_column_indices(ci, columns);
            auto vals = mx::array(values.data(), {int(values.size())}, mx::float32);
            auto vectors = mx::array(x.data(), {columns,padded}, mx::float32);
            auto mask = mx::array(active.data(), {padded}, mx::int32);
            for (int reduction : {1,32/tile}) {
                auto product = detail::metal_spmm(starts,indices,vals,vectors,mask,rows,tile,reduction,stream);
                mx::eval(product);
                for (int r = 0; r < rows; ++r) for (int j = 0; j < padded; ++j) {
                    double expected = 0;
                    if (j < width) for (int k = rp[r]; k < rp[r+1]; ++k)
                        expected += double(values[k])*x[size_t(ci[k])*padded+j];
                    double actual = product.data<float>()[r*padded+j];
                    if (!std::isfinite(actual) || std::abs(actual-expected) > 1e-5*(1+std::abs(expected)))
                        throw std::runtime_error("SpMM disagrees with original FP64 product");
                }
            }
            // Build CSR(A^T), then verify against a host scatter using A.
            // In particular A has long rows while A^T has many empty/short
            // rows: each orientation must admit its own row schedule.
            std::vector<int32_t> trp(columns+1,0), tci(ci.size());
            std::vector<float> tv(values.size());
            for (int col : ci) ++trp[col+1];
            for (int col=0;col<columns;++col) trp[col+1]+=trp[col];
            auto next=trp;
            for (int r=0;r<rows;++r) for (int a=rp[r];a<rp[r+1];++a) {
                const int offset=next[ci[a]]++;
                tci[offset]=r;tv[offset]=values[a];
            }
            std::vector<float> y(rows*padded,NAN);
            std::vector<double> expected(size_t(columns)*padded,0);
            for (int r=0;r<rows;++r) for (int j=0;j<width;++j) {
                y[r*padded+j]=float(int(rng()%23)-11)/8;
                for (int a=rp[r];a<rp[r+1];++a)
                    expected[size_t(ci[a])*padded+j]+=double(values[a])*y[r*padded+j];
            }
            auto ts=mx::array(trp.data(),{columns+1},mx::int32);
            auto ti=detail::metal_column_indices(tci,rows);
            auto ta=mx::array(tv.data(),{int(tv.size())},mx::float32);
            auto ty=mx::array(y.data(),{rows,padded},mx::float32);
            for (int reduction : {1,32/tile}) {
                auto product=detail::metal_spmm(ts,ti,ta,ty,mask,columns,tile,reduction,stream);
                mx::eval(product);
                for (size_t a=0;a<expected.size();++a)
                    if (!std::isfinite(product.data<float>()[a]) ||
                        std::abs(product.data<float>()[a]-expected[a])>1e-5*(1+std::abs(expected[a])))
                        throw std::runtime_error("transpose SpMM disagrees with original FP64 scatter");
            }
        }
        std::cout << "Metal SpMM widths/tails/empty/duplicates/long rows/index boundary passed\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
