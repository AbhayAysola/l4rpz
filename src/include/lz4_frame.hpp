#pragma once

#include "lz4.hpp"

constexpr uint32_t FRAME_MAGIC_NUMBER = 0x184D2204;

LZ4FrameConfig parse_frame_header(const LZ4FrameHeaderRaw &raw);