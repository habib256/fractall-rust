#!/bin/bash
# Fraktaler 3 -- fast deep escape time fractals
# Copyright (C) 2021-2025 Claude Heiland-Allen
# SPDX-License-Identifier: AGPL-3.0-only

set -e
TOP="$(pwd)"
prefix="${HOME}/opt/android/21"
make headers
VERSION="$(cat VERSION.txt | head -n 1)"
SDLVERSION=2.32.8
if [[ "$1" =~ "prepare" ]]
then
mkdir -p "${TOP}/android/src"
cd "${TOP}/src"
mkdir -p arm64-v8a armeabi-v7a x86 x86_64
for d in lib include
do
ln -fs "${prefix}/aarch64/$d/" "${TOP}/src/arm64-v8a/$d"
ln -fs "${prefix}/armv7a/$d/" "${TOP}/src/armeabi-v7a/$d"
ln -fs "${prefix}/i686/$d/" "${TOP}/src/x86/$d"
ln -fs "${prefix}/x86_64/$d/" "${TOP}/src/x86_64/$d"
done
ln -fs ../../imgui/
ln -fs ../../imgui-filebrowser/
ln -fs ../../implot/
ln -fs ../../fraktaler-3/
cd "${TOP}/android/src"
wget -c https://github.com/libsdl-org/SDL/releases/download/release-${SDLVERSION}/SDL2-${SDLVERSION}.tar.gz
tar xaf SDL2-${SDLVERSION}.tar.gz
cd SDL2-${SDLVERSION}/build-scripts
./androidbuild.sh uk.co.mathr.fraktaler.v3 ../../../../src/main.cc
cd ../build/uk.co.mathr.fraktaler.v3/
rm -rf app
ln -fs "${TOP}/app" app
mkdir -p app/jni/SDL
ln -fs "${TOP}/src" app/jni/src
ln -fs "${TOP}/android/src/SDL2-${SDLVERSION}/include" app/jni/SDL/include
ln -fs "${TOP}/android/src/SDL2-${SDLVERSION}/src" app/jni/SDL/src
ln -fs "${TOP}/android/src/SDL2-${SDLVERSION}/android-project/app/src/main/java/org" app/src/main/java/org
else
cd "${TOP}/android/src/SDL2-${SDLVERSION}/build/uk.co.mathr.fraktaler.v3"
sed "s|VERSION|${VERSION}|g" < app/build.gradle.in > app/build.gradle
if [[ "$1" =~ "release" ]]
then
  ./gradlew assembleRelease
  cd app/build/outputs/apk/release/
  zipalign -v -p 4 app-release-unsigned.apk app-release-unsigned-aligned.apk
  apksigner sign --ks ~/.fraktaler-3.ks --out "uk.co.mathr.fraktaler.v3-${VERSION}.apk" app-release-unsigned-aligned.apk
  cp -avi "uk.co.mathr.fraktaler.v3-${VERSION}.apk" "${TOP}"
  adb install -r "uk.co.mathr.fraktaler.v3-${VERSION}.apk"
else
  ./gradlew installDebug
fi
fi
