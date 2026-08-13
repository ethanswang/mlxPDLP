#!/usr/bin/env bash
set -euo pipefail

data_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${data_dir}/../../.." && pwd)"
inspector="${repo_dir}/build/mlxpdlp_mps_inspect"
manifest="${data_dir}/manifest.tsv"
known_differences="${data_dir}/known_differences.tsv"
results="${data_dir}/dimensions.tsv"

if [[ ! -x "${inspector}" ]]; then
    echo "Missing ${inspector}; configure with -DMLXPDLP_BUILD_BENCHMARKS=ON" >&2
    exit 1
fi

printf 'name\tpublished_rows\tpublished_columns\tpublished_nonzeros\tconstraint_rows\tcolumns\tmatrix_nonzeros\tobjective_nonzeros\tstatus\n' >"${results}"
failures=0
while IFS=$'\t' read -r name rows columns nonzeros format url; do
    if [[ "${name}" == "name" || -z "${name}" ]]; then
        continue
    fi
    model="${data_dir}/${name}.mps.gz"
    if [[ ! -s "${model}" ]]; then
        printf '%s\t%s\t%s\t%s\t-\t-\t-\t-\tMISSING\n' \
            "${name}" "${rows}" "${columns}" "${nonzeros}" >>"${results}"
        failures=$((failures + 1))
        continue
    fi

    echo "Inspecting ${name}"
    parsed="$("${inspector}" "${model}" | tail -n 1)"
    IFS=$'\t' read -r path actual_rows actual_columns actual_nonzeros objective_nonzeros <<<"${parsed}"
    status="OK"
    if [[ "${actual_rows}" != "${rows}" || "${actual_columns}" != "${columns}" ||
          "${actual_nonzeros}" != "${nonzeros}" ]]; then
        rows_match=false
        nonzeros_match=false
        if [[ "${actual_rows}" == "${rows}" || $((actual_rows + 1)) == "${rows}" ]]; then
            rows_match=true
        fi
        if [[ "${actual_nonzeros}" == "${nonzeros}" ||
              $((actual_nonzeros + objective_nonzeros)) == "${nonzeros}" ]]; then
            nonzeros_match=true
        fi

        if [[ "${actual_columns}" == "${columns}" &&
              "${rows_match}" == true && "${nonzeros_match}" == true ]]; then
            status="OK_PUBLISHED_CONVENTION"
        else
            known="$(awk -F $'\t' -v target="${name}" \
                '$1 == target { print $2 "\t" $3 "\t" $4; exit }' "${known_differences}")"
            IFS=$'\t' read -r known_rows known_columns known_nonzeros <<<"${known}"
            if [[ "${actual_rows}" == "${known_rows:-}" &&
                  "${actual_columns}" == "${known_columns:-}" &&
                  "${actual_nonzeros}" == "${known_nonzeros:-}" ]]; then
                status="KNOWN_SOURCE_DIFFERENCE"
            else
                status="MISMATCH"
                failures=$((failures + 1))
            fi
        fi
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${name}" "${rows}" "${columns}" "${nonzeros}" \
        "${actual_rows}" "${actual_columns}" "${actual_nonzeros}" \
        "${objective_nonzeros}" "${status}" >>"${results}"
done <"${manifest}"

if [[ "${failures}" -ne 0 ]]; then
    echo "${failures} LPfeas files are missing or have dimension mismatches; see ${results}" >&2
    exit 1
fi
echo "All public LPfeas dimensions match; see ${results}"
