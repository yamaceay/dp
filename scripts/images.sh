#!/bin/bash

rg 'image\(' docs/thesis -g '*.typ' | 
sed -En 's|.*image\("([^",)]*)".*|\1|p' | 
grep -E '^(\.\./)?images/' | 
sort -u | 
while IFS= read -r p; do 
    src="docs/thesis/${p#../}"
    dst="${src/images/images_mini}"
    if [ -d "$dst" ]; then
        rm -rf -- "$dst"
    fi
    mkdir -p "$(dirname -- "$dst")" && cp -- "$src" "$dst"
done