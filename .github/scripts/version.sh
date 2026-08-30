#!/usr/bin/env bash
# Calcule la version de distribution. SOURCÉ par package.sh et appimage.sh pour
# que le tag/sha d'un même build produise le même nom dans tous les artefacts.
#
# Le tag fait foi lors d'une release ; sinon version du Cargo.toml suffixée du
# sha court, pour que deux builds manuels ne se confondent pas.
fractall_version() {
  if [[ "${GITHUB_REF_TYPE:-}" == "tag" && -n "${GITHUB_REF_NAME:-}" ]]; then
    echo "${GITHUB_REF_NAME#v}"
    return
  fi
  local cargo_version sha
  cargo_version="$(sed -n 's/^version *= *"\(.*\)"/\1/p' Cargo.toml | head -1)"
  # `GITHUB_SHA` d'abord : dans un conteneur tournant en root, `git rev-parse`
  # échoue sur « dubious ownership » du checkout (artefacts versionnés
  # `gunknown`). Repli sur git en local, hors CI.
  if [[ -n "${GITHUB_SHA:-}" ]]; then
    sha="${GITHUB_SHA:0:7}"
  else
    sha="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  fi
  echo "${cargo_version}-g${sha}"
}
