#!/bin/bash
# 측정 전에 폴더 미리 만들어 두는 스크립트.
# 원하는 위치에서 실행하면 s01..s10 폴더 + condition 하위 폴더 생성.

ROOT="${1:-./H-Walker_pilot_2026}"
mkdir -p "$ROOT"
cd "$ROOT"
mkdir -p s01
mkdir -p s02
mkdir -p s03
mkdir -p s04
mkdir -p s05
mkdir -p s06
mkdir -p s07
mkdir -p s08
mkdir -p s09
mkdir -p s10

echo "폴더 생성 완료: $ROOT/"
ls -1