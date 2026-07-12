#!/usr/bin/env bash
# downloads the silesia corpus and compresses it into data/
# data/silesia*/ is .gitignored — run this once before using --silesia
set -euo pipefail

mkdir -p data
cd data

echo "downloading silesia corpus..."
curl -L -o silesia.zip "http://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip"

echo "extracting..."
unzip -q silesia.zip -d silesia
rm silesia.zip

echo "compressing (default blocks)..."
mkdir -p silesia_lz4
for f in silesia/*; do
    lz4 -q -f "$f" "silesia_lz4/$(basename "$f").lz4"
done

echo "compressing (-B4, 64KB blocks)..."
mkdir -p silesia_lz4_b4
for f in silesia/*; do
    lz4 -q -f -B4 "$f" "silesia_lz4_b4/$(basename "$f").lz4"
done

echo "done. run: python tests/harness.py --silesia"
