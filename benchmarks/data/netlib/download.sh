#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
source_dir="$script_dir/sources"
emps="$script_dir/../lpfeas/tools/emps"

if [ ! -x "$emps" ]; then
    echo "missing EMPS expander: $emps" >&2
    exit 1
fi

mkdir -p "$source_dir"

cases="
afiro
kb2
sc50b
sc50a
adlittle
blend
sc105
recipe
share2b
stocfor1
scagr7
boeing2
lotfi
israel
share1b
forplan
beaconfd
vtp.base
sc205
brandy
e226
bore3d
capri
bandm
sctap1
grow15
scfxm1
tuff
boeing1
stair
standata
scorpion
scsd8
etamacro
degen2
finnis
bnl1
25fv47
fit1d
pilot87
"

for name in $cases; do
    source_file="$source_dir/$name"
    output_file="$script_dir/$name.mps.gz"
    if [ ! -f "$source_file" ]; then
        curl -fsSLo "$source_file" "https://www.netlib.org/lp/data/$name"
    fi
    if [ ! -f "$output_file" ]; then
        "$emps" < "$source_file" | gzip -9 -n > "$output_file"
    fi
done

echo "Netlib small-to-medium corpus is ready in $script_dir"
