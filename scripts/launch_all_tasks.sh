#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
group="${GANTRY_GROUP:-novomol-$(date -u +%Y%m%d-%H%M%S)}"

tasks=(
  "Albuterol_Similarity"
  "Amlodipine_MPO"
  "Celecoxib_Rediscovery"
  "DRD2"
  "Deco Hop"
  "Fexofenadine_MPO"
  "GSK3B"
  "Isomers_C7H8N2O2"
  "Isomers_C9H10N2O2PF2Cl"
  "JNK3"
  "Median 1"
  "Median 2"
  "Mestranol_Similarity"
  "Osimertinib_MPO"
  "Perindopril_MPO"
  "QED"
  "Ranolazine_MPO"
  "Scaffold Hop"
  "Sitagliptin_MPO"
  "Thiothixene_Rediscovery"
  "Troglitazone_Rediscovery"
  "Zaleplon_MPO"
)

for task in "${tasks[@]}"; do
  GANTRY_GROUP="$group" "$repo_root/scripts/run_gantry.sh" "$task" "$@"
done
