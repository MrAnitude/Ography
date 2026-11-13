#include <gtest/gtest.h>

#include "engine/stop_checker.h"

// TODO: stop string split across two tokens is not emitted; EOS id finishes with Stop;
//       max_tokens finishes with Length; held-back text is released when ruled out.
TEST(StopChecker, StopStringAcrossTokens) { GTEST_SKIP() << "not implemented yet"; }
