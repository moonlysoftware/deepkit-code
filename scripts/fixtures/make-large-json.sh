#!/usr/bin/env sh
set -eu

bytes=${1:?expected payload byte count}
printf '{"payload":"'
head -c "$bytes" /dev/zero | tr '\\0' x
printf '"}\n'
