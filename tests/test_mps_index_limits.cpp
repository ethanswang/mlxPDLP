// Copyright 2026 Ethan Wang <ethanshurui.wang@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#include "mlxPDLP/mps_loader.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>

// This target compiles the real parser with an index limit of eight so the
// count boundaries can be exercised through MPS files without huge allocations.
static bool check_model(int rows, int columns, int nonzeros, bool accepted) {
    std::string input = "NAME LIMITS\nROWS\n N OBJ\n";
    for (int row = 0; row < rows; ++row)
        input += " E R" + std::to_string(row) + "\n";
    input += "COLUMNS\n";
    for (int col = 0; col < columns; ++col)
        input += " X" + std::to_string(col) + " OBJ 1\n";
    for (int entry = 0; entry < nonzeros; ++entry)
        input += " X0 R" + std::to_string(entry % rows) + " 1\n";
    // Reusing a name at the variable-count limit must remain valid.
    input += " X" + std::to_string(columns - 1) + " OBJ 2\nENDATA\n";

    char path[] = "/tmp/mlxpdlp-index-limits-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0)
        return false;
    FILE *file = fdopen(fd, "w");
    if (!file) {
        close(fd);
        std::remove(path);
        return false;
    }
    const bool written = std::fwrite(input.data(), 1, input.size(), file) == input.size();
    std::fclose(file);
    auto *problem = written ? mlxpdlp_mps_problem_load(path) : nullptr;
    std::remove(path);
    const bool passed = written && (accepted
        ? problem && problem->num_constraints == rows &&
              problem->num_variables == columns && problem->num_nonzeros == nonzeros &&
              problem->row_ptr[rows] == nonzeros &&
              problem->objective[columns - 1] == 3.0
        : problem == nullptr);
    mlxpdlp_mps_problem_free(problem);
    if (!passed)
        std::fprintf(stderr, "FAIL: rows=%d columns=%d nonzeros=%d expected %s\n",
                     rows, columns, nonzeros, accepted ? "accept" : "reject");
    return passed;
}

int main() {
    bool passed = true;
    passed &= check_model(8, 8, 8, true);
    passed &= check_model(1, 8, 8, true);
    passed &= check_model(8, 1, 8, true);
    passed &= check_model(1, 1, 8, true);
    passed &= check_model(1, 1, 0, true);
    passed &= check_model(9, 1, 1, false);
    passed &= check_model(1, 9, 1, false);
    passed &= check_model(1, 1, 9, false);
    return passed ? 0 : 1;
}
