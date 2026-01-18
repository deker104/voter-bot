#!/bin/bash

mkdir -p assets

for i in {1..15}; do
    magick -size 500x500 xc:white \
    -gravity center \
    -pointsize 200 \
    -fill black \
    -annotate +0+0 "$i" \
    "assets/$i.png"
done

echo "Готово! Картинки сохранены в папке assets"