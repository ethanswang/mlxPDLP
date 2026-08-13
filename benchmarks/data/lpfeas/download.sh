#!/usr/bin/env bash
set -euo pipefail

data_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
manifest="${data_dir}/manifest.tsv"
source_dir="${data_dir}/sources"
tool_dir="${data_dir}/tools"
mkdir -p "${source_dir}" "${tool_dir}"

emps_source="${source_dir}/emps.c"
emps_binary="${tool_dir}/emps"
if [[ ! -x "${emps_binary}" ]]; then
    curl -fL --retry 5 --retry-delay 2 \
        -o "${emps_source}.part" \
        https://www.netlib.org/lp/data/emps.c
    mv "${emps_source}.part" "${emps_source}"
    cc -O2 "${emps_source}" -o "${emps_binary}"
fi

download_archive() {
    local url="$1"
    local archive="$2"
    if [[ -s "${archive}" ]]; then
        return
    fi
    echo "Downloading ${url}"
    curl -fL --retry 5 --retry-delay 2 --continue-at - \
        -o "${archive}.part" "${url}"
    mv "${archive}.part" "${archive}"
}

while IFS=$'\t' read -r name rows columns nonzeros format url; do
    if [[ "${name}" == "name" || -z "${name}" ]]; then
        continue
    fi

    output="${data_dir}/${name}.mps.gz"
    if [[ -s "${output}" ]] && gzip -t "${output}" 2>/dev/null; then
        echo "Ready ${name}"
        continue
    fi

    archive_name="${url##*/}"
    archive="${source_dir}/${archive_name}"
    download_archive "${url}" "${archive}"

    echo "Expanding ${name} (${format})"
    output_part="${output}.part"
    case "${format}" in
    mps_bz2)
        bzip2 -dc "${archive}" | gzip -6 >"${output_part}"
        ;;
    emps_bz2)
        bzip2 -dc "${archive}" | "${emps_binary}" - | gzip -6 >"${output_part}"
        ;;
    mps_gz)
        gzip -t "${archive}"
        ln -f "${archive}" "${output_part}"
        ;;
    emps_gz)
        gzip -dc "${archive}" | "${emps_binary}" - | gzip -6 >"${output_part}"
        ;;
    *)
        echo "Unknown source format '${format}' for ${name}" >&2
        exit 1
        ;;
    esac
    gzip -t "${output_part}"
    mv "${output_part}" "${output}"
done <"${manifest}"

echo "LPfeas public data is ready in ${data_dir}"
