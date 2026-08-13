/*
Copyright 2026 Ethan Wang <ethanshurui.wang@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/*
Minimal test to verify MLX matmul and basic ops work correctly.
*/
#include "mlx/mlx.h"
#include <cmath>
#include <cstdio>

namespace mx = mlx::core;

int main() {
    // This legacy diagnostic uses float64 arrays, which MLX executes on CPU.
    // Metal coverage lives in test_device_comparison.cpp using float32.
    mx::set_default_device(mx::Device::cpu);

    printf("MLX Basic Diagnostic Test\n");
    printf("=========================\n\n");

    // Test 1: Create arrays
    printf("Test 1: Array creation\n");
    double a_data[] = {1.0, 2.0, 0.0, 1.0, 3.0, 2.0};
    auto A = mx::array(a_data, {3, 2}, mx::float64);
    auto x = mx::ones({2}, mx::float64);
    mx::eval(A, x);
    printf("  A = [3x2] created OK\n");
    printf("  x = [2] ones created OK\n");

    // Test 2: matmul
    printf("\nTest 2: matmul A * x\n");
    auto x_2d = mx::reshape(x, {2, 1});
    auto Ax_2d = mx::matmul(A, x_2d);
    auto Ax = mx::reshape(Ax_2d, {3});
    mx::eval(Ax);
    printf("  A*x data: ");
    auto Ax_ptr = Ax.data<double>();
    for (int i = 0; i < 3; i++)
        printf("%.6f ", Ax_ptr[i]);
    printf("\n  Expected:  3.000000 1.000000 5.000000\n");

    // Test 3: matmul A^T * y
    printf("\nTest 3: matmul A^T * y\n");
    auto AT = mx::transpose(A);
    mx::eval(AT);
    auto y = mx::array(new double[3]{1.0, 1.0, 1.0}, {3}, mx::float64,
                       [](void *p) { delete[] (double *)p; });
    mx::eval(y);
    auto y_2d = mx::reshape(y, {3, 1});
    auto ATy_2d = mx::matmul(AT, y_2d);
    auto ATy = mx::reshape(ATy_2d, {2});
    mx::eval(ATy);
    printf("  A^T*y data: ");
    auto ATy_ptr = ATy.data<double>();
    for (int i = 0; i < 2; i++)
        printf("%.6f ", ATy_ptr[i]);
    printf("\n  Expected:    4.000000 5.000000\n");

    // Test 4: L2 norm
    printf("\nTest 4: L2 norm\n");
    auto n = mx::linalg::norm(Ax);
    mx::eval(n);
    printf("  ||Ax||_2 = %.6f (expected: sqrt(9+1+25)=%.6f)\n", n.item<double>(), std::sqrt(35.0));

    // Test 5: dot product (same-shaped vectors)
    printf("\nTest 5: Dot product\n");
    auto v1 = mx::array(new double[3]{1.0, 2.0, 3.0}, {3}, mx::float64,
                        [](void *p) { delete[] (double *)p; });
    auto v2 = mx::array(new double[3]{4.0, 5.0, 6.0}, {3}, mx::float64,
                        [](void *p) { delete[] (double *)p; });
    mx::eval(v1, v2);
    auto dot_v = mx::sum(v1 * v2);
    mx::eval(dot_v);
    printf("  dot([1,2,3], [4,5,6]) = %.6f (expected: 32)\n", dot_v.item<double>());

    // Test 6: Power method on this small matrix
    printf("\nTest 6: Power method for ||A||_2\n");
    auto eigen = mx::ones({3}, mx::float64);
    for (int iter = 0; iter < 100; ++iter) {
        // Normalize
        auto en_arr = mx::linalg::norm(eigen);
        mx::eval(en_arr);
        double en = en_arr.item<double>();
        if (en < 1e-14) {
            printf("  [iter %d] norm is zero, breaking\n", iter);
            break;
        }
        eigen = eigen / en;

        // y = AT * eigen
        auto eig_2d = mx::reshape(eigen, {3, 1});
        auto y_temp = mx::matmul(AT, eig_2d);
        auto y_1d = mx::reshape(y_temp, {2});
        mx::eval(y_1d);

        // eigen_new = A * y
        auto y_2d_2 = mx::reshape(y_1d, {2, 1});
        auto eig_new_2d = mx::matmul(A, y_2d_2);
        auto eigen_new = mx::reshape(eig_new_2d, {3});
        mx::eval(eigen_new);

        double sigma_sq = mx::sum(eigen * eigen_new).item<double>();
        auto residual = eigen_new - sigma_sq * eigen;
        double rn = mx::linalg::norm(residual).item<double>();

        if (iter < 5 || rn < 1e-6) {
            printf("  iter %3d: sigma_sq=%.6f sigma=%.6f residual=%.6e\n", iter, sigma_sq,
                   std::sqrt(std::fabs(sigma_sq)), rn);
        }
        eigen = eigen_new;
        if (rn < 1e-6) {
            printf("  Converged!\n");
            break;
        }
    }

    // Test 7: clip with infinities
    printf("\nTest 7: clip with infinities\n");
    double inf = std::numeric_limits<double>::infinity();
    auto temp = mx::array(new double[2]{-100.0, 100.0}, {2}, mx::float64,
                          [](void *p) { delete[] (double *)p; });
    auto lb = mx::array(new double[2]{0.0, -inf}, {2}, mx::float64,
                        [](void *p) { delete[] (double *)p; });
    auto ub = mx::array(new double[2]{inf, 50.0}, {2}, mx::float64,
                        [](void *p) { delete[] (double *)p; });
    mx::eval(temp, lb, ub);
    auto clipped = mx::clip(temp, lb, ub);
    mx::eval(clipped);
    auto c_ptr = clipped.data<double>();
    printf("  clip([-100, 100], [0, -inf], [inf, 50]) = [%.1f, %.1f] (expected: [0, 50])\n",
           c_ptr[0], c_ptr[1]);

    printf("\nAll diagnostics complete.\n");
    return 0;
}
