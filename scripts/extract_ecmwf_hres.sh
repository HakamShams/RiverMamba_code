#!/bin/sh

# you can change this to your absolute path
dir="./ECMWF_HRES_Global"

find "$dir" -name '*.7z.001' -exec 7za x {} -o"$dir" \;