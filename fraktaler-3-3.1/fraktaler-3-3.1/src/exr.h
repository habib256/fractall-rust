// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#ifdef HAVE_EXR
#if HAVE_EXR == 0
#undef HAVE_EXR
#else
#if HAVE_EXR == 2

#include <ImfNamespace.h>
#include <ImfOutputFile.h>
#include <ImfHeader.h>
#include <ImfChannelList.h>
#include <ImfFloatAttribute.h>
#include <ImfIntAttribute.h>
#include <ImfStringAttribute.h>
#include <ImfArray.h>
#include <ImfFrameBuffer.h>
#include <ImfMultiPartInputFile.h>
namespace IMF = OPENEXR_IMF_NAMESPACE;
using namespace IMF;
using namespace IMATH_NAMESPACE;

#else
#if HAVE_EXR == 3

#include <Imath/ImathBox.h>
#include <OpenEXR/ImfThreading.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfStringAttribute.h>
#include <OpenEXR/ImfFloatAttribute.h>
#include <OpenEXR/ImfIntAttribute.h>
#include <OpenEXR/ImfPixelType.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfMultiPartInputFile.h>
namespace IMF = OPENEXR_IMF_NAMESPACE;
using namespace IMF;
using namespace IMATH_NAMESPACE;

#else
#error unsupported OpenEXR version (HAVE_EXR supported values: 2, 3)
#endif
#endif
#endif
#endif
