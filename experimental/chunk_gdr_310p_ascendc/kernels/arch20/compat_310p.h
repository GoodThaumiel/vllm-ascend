#ifndef COMPAT_310P_H
#define COMPAT_310P_H

#ifndef __CCE_KT_TEST__
#include "kernel_operator.h"
#endif

// CANN 9.2.1 BiSheng provides bfloat16_t for dav_m200.
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)
#define __COMPAT_310P_ACTIVE__
#endif

// 310P has no fixpipe unit; post-matmul stores go through MTE3
#ifndef PIPE_FIX
#define PIPE_FIX PIPE_MTE3
#endif

// 310P renames LoadDataWithSparse → LoadDataWithSparseCal
#ifdef __COMPAT_310P_ACTIVE__
#define LoadDataWithSparse LoadDataWithSparseCal
#endif

#endif // COMPAT_310P_H
