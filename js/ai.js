/**
 * 2048 AI —— 基于 nneonneo/2048-ai 的 expectimax 算法
 *
 * 内部棋盘表示：4 个 16-bit Number（每行一个），与 C++ 的 row_t 完全一致。
 * 所有搜索操作为纯 Number 位运算，速度接近原生。
 * BigInt 仅用于 init_tables 中构建列移动表，运行时完全不涉及。
 *
 * 棋盘布局（与 C++ 一致）：
 *   每行 4 个 nibble (4-bit)，nibble 0=col0, nibble 1=col1, ...
 *   row[0] 是第 0 行，row[3] 是第 3 行
 *
 * 方向映射（C++ → 游戏）：
 *   C++: 0=up, 1=down, 2=left, 3=right
 *   游戏: 0=up, 1=right, 2=down, 3=left
 *   C++ → 游戏: [0, 2, 3, 1]
 */

(function () {
  'use strict';

  // ============================================================
  // 常量
  // ============================================================

  // ---- 启发式评分权重（与 C++ 完全一致） ----
  const SCORE_LOST_PENALTY = 200000.0;
  const SCORE_MONOTONICITY_POWER = 4.0;
  const SCORE_MONOTONICITY_WEIGHT = 47.0;
  const SCORE_SUM_POWER = 3.5;
  const SCORE_SUM_WEIGHT = 11.0;
  const SCORE_MERGES_WEIGHT = 700.0;
  const SCORE_EMPTY_WEIGHT = 270.0;

  // ---- 搜索参数 ----
  /** 累计概率低于此阈值时剪枝 */
  const CPROB_THRESH_BASE = 0.0001;
  /** 置换表缓存深度上限 */
  const CACHE_DEPTH_LIMIT = 15;

  // ============================================================
  // 预计算查找表（65536 条，按 16-bit 行索引）
  // ============================================================

  // ---- 移动表（16-bit Number） ----
  const row_left_table = new Uint16Array(65536);
  const row_right_table = new Uint16Array(65536);
  // 列移动表：拆分为每行 16-bit，避免 BigInt 运算
  const col_up_r0 = new Uint16Array(65536);
  const col_up_r1 = new Uint16Array(65536);
  const col_up_r2 = new Uint16Array(65536);
  const col_up_r3 = new Uint16Array(65536);
  const col_down_r0 = new Uint16Array(65536);
  const col_down_r1 = new Uint16Array(65536);
  const col_down_r2 = new Uint16Array(65536);
  const col_down_r3 = new Uint16Array(65536);

  // ---- 评分表 ----
  const heur_score_table = new Float32Array(65536);
  const score_table = new Float32Array(65536);

  // ---- 辅助表 ----
  /** 每行的空格计数 */
  const empty_count_table = new Uint8Array(65536);

  /** 查找表是否已初始化 */
  let tablesInitialized = false;

  // ---- 预计算常量（避免热路径中重复计算） ----
  /** tile 2 的 nibble 值按列位置：[1, 16, 256, 4096] */
  const TILE2_NIB = [1, 1 << 4, 1 << 8, 1 << 12];
  /** tile 4 的 nibble 值按列位置：[2, 32, 512, 8192] */
  const TILE4_NIB = [2, 2 << 4, 2 << 8, 2 << 12];

  // ============================================================
  // 底层操作
  // ============================================================

  /**
   * 反转一行中 4 个 nibble 的顺序：[a,b,c,d] → [d,c,b,a]
   * & 0xFFFF 截断为 16 位，模拟 C++ 中 row_t (uint16_t) 行为
   */
  function reverse_row(row) {
    return ((row >> 12) | ((row >> 4) & 0x00f0) | ((row << 4) & 0x0f00) | (row << 12)) & 0xFFFF;
  }

  // ============================================================
  // 初始化查找表
  // ============================================================

  /**
   * 将 16-bit 行值展开为 64-bit 列掩码（仅 init 阶段用 BigInt）
   */
  function unpack_col_bi(r) {
    const t = BigInt(r);
    return (t | (t << 12n) | (t << 24n) | (t << 36n)) & 0x000F000F000F000Fn;
  }

  function init_tables() {
    if (tablesInitialized) return;

    for (let row = 0; row < 65536; row++) {
      const line = [
        (row >> 0) & 0xf,
        (row >> 4) & 0xf,
        (row >> 8) & 0xf,
        (row >> 12) & 0xf,
      ];

      // ---- 空格计数 ----
      let ec = 0;
      for (let i = 0; i < 4; i++) {
        if (line[i] === 0) ec++;
      }
      empty_count_table[row] = ec;

      // ---- 实际得分 ----
      let score = 0.0;
      for (let i = 0; i < 4; i++) {
        const rank = line[i];
        if (rank >= 2) {
          score += (rank - 1) * (1 << rank);
        }
      }
      score_table[row] = score;

      // ---- 启发式评分 ----
      let sum = 0;
      let empty = 0;
      let merges = 0;
      let prev = 0;
      let counter = 0;

      for (let i = 0; i < 4; i++) {
        const rank = line[i];
        sum += Math.pow(rank, SCORE_SUM_POWER);
        if (rank === 0) {
          empty++;
        } else {
          if (prev === rank) {
            counter++;
          } else if (counter > 0) {
            merges += 1 + counter;
            counter = 0;
          }
          prev = rank;
        }
      }
      if (counter > 0) {
        merges += 1 + counter;
      }

      let monotonicity_left = 0;
      let monotonicity_right = 0;
      for (let i = 1; i < 4; i++) {
        if (line[i - 1] > line[i]) {
          monotonicity_left +=
            Math.pow(line[i - 1], SCORE_MONOTONICITY_POWER) -
            Math.pow(line[i], SCORE_MONOTONICITY_POWER);
        } else {
          monotonicity_right +=
            Math.pow(line[i], SCORE_MONOTONICITY_POWER) -
            Math.pow(line[i - 1], SCORE_MONOTONICITY_POWER);
        }
      }

      heur_score_table[row] =
        SCORE_LOST_PENALTY +
        SCORE_EMPTY_WEIGHT * empty +
        SCORE_MERGES_WEIGHT * merges -
        SCORE_MONOTONICITY_WEIGHT * Math.min(monotonicity_left, monotonicity_right) -
        SCORE_SUM_WEIGHT * sum;

      // ---- 模拟左移 ----
      const moved = [line[0], line[1], line[2], line[3]];
      for (let i = 0; i < 3; i++) {
        let j;
        for (j = i + 1; j < 4; j++) {
          if (moved[j] !== 0) break;
        }
        if (j === 4) break;
        if (moved[i] === 0) {
          moved[i] = moved[j];
          moved[j] = 0;
          i--;
        } else if (moved[i] === moved[j]) {
          if (moved[i] !== 0xf) moved[i]++;
          moved[j] = 0;
        }
      }

      const result =
        (moved[0] << 0) | (moved[1] << 4) | (moved[2] << 8) | (moved[3] << 12);
      const rev_result = reverse_row(result);
      const rev_row = reverse_row(row);

      // 行移动表（纯 Number）
      row_left_table[row] = row ^ result;
      row_right_table[rev_row] = rev_row ^ rev_result;

      // 列移动表：用 BigInt 构建一次，然后拆成 4×16-bit 存入
      // col_up 拆分
      const cu = unpack_col_bi(row) ^ unpack_col_bi(result);
      col_up_r0[row] = Number((cu >> 0n) & 0xFFFFn);
      col_up_r1[row] = Number((cu >> 16n) & 0xFFFFn);
      col_up_r2[row] = Number((cu >> 32n) & 0xFFFFn);
      col_up_r3[row] = Number((cu >> 48n) & 0xFFFFn);

      // col_down 拆分
      const cd = unpack_col_bi(rev_row) ^ unpack_col_bi(rev_result);
      col_down_r0[rev_row] = Number((cd >> 0n) & 0xFFFFn);
      col_down_r1[rev_row] = Number((cd >> 16n) & 0xFFFFn);
      col_down_r2[rev_row] = Number((cd >> 32n) & 0xFFFFn);
      col_down_r3[rev_row] = Number((cd >> 48n) & 0xFFFFn);
    }

    tablesInitialized = true;
  }

  // ============================================================
  // 行数组 → 字符串键（用于置换表 Map）
  // 字符串拼接比 BigInt 位运算更快，且 Map<string, ...> 哈希查找高效
  // ============================================================

  function rows_to_key(rows) {
    // 利用 4 个 16-bit 值直接拼成字符串，无需 BigInt 分配
    return '' + rows[0] + ',' + rows[1] + ',' + rows[2] + ',' + rows[3];
  }

  // ============================================================
  // 执行移动（纯 Number 运算，返回新数组）
  // ============================================================

  function rows_equal(a, b) {
    return a[0] === b[0] && a[1] === b[1] && a[2] === b[2] && a[3] === b[3];
  }

  /** 上移（C++ move 0）—— 内联列提取，逐列处理 */
  function move_up(rows) {
    const r = rows;
    const n0 = r[0], n1 = r[1], n2 = r[2], n3 = r[3];
    const new_rows = [n0, n1, n2, n3];

    // col 0: 内联提取 + 应用
    let cv = (n0 & 0xf) | ((n1 & 0xf) << 4) | ((n2 & 0xf) << 8) | ((n3 & 0xf) << 12);
    new_rows[0] ^= col_up_r0[cv];
    new_rows[1] ^= col_up_r1[cv];
    new_rows[2] ^= col_up_r2[cv];
    new_rows[3] ^= col_up_r3[cv];

    // col 1
    cv = ((n0 >> 4) & 0xf) | (((n1 >> 4) & 0xf) << 4) | (((n2 >> 4) & 0xf) << 8) | (((n3 >> 4) & 0xf) << 12);
    new_rows[0] ^= col_up_r0[cv] << 4;
    new_rows[1] ^= col_up_r1[cv] << 4;
    new_rows[2] ^= col_up_r2[cv] << 4;
    new_rows[3] ^= col_up_r3[cv] << 4;

    // col 2
    cv = ((n0 >> 8) & 0xf) | (((n1 >> 8) & 0xf) << 4) | (((n2 >> 8) & 0xf) << 8) | (((n3 >> 8) & 0xf) << 12);
    new_rows[0] ^= col_up_r0[cv] << 8;
    new_rows[1] ^= col_up_r1[cv] << 8;
    new_rows[2] ^= col_up_r2[cv] << 8;
    new_rows[3] ^= col_up_r3[cv] << 8;

    // col 3
    cv = ((n0 >> 12) & 0xf) | (((n1 >> 12) & 0xf) << 4) | (((n2 >> 12) & 0xf) << 8) | (((n3 >> 12) & 0xf) << 12);
    new_rows[0] ^= col_up_r0[cv] << 12;
    new_rows[1] ^= col_up_r1[cv] << 12;
    new_rows[2] ^= col_up_r2[cv] << 12;
    new_rows[3] ^= col_up_r3[cv] << 12;

    new_rows[0] &= 0xFFFF; new_rows[1] &= 0xFFFF;
    new_rows[2] &= 0xFFFF; new_rows[3] &= 0xFFFF;
    return new_rows;
  }

  /** 下移（C++ move 1）—— 内联列提取，逐列处理 */
  function move_down(rows) {
    const r = rows;
    const n0 = r[0], n1 = r[1], n2 = r[2], n3 = r[3];
    const new_rows = [n0, n1, n2, n3];

    // col 0
    let cv = (n0 & 0xf) | ((n1 & 0xf) << 4) | ((n2 & 0xf) << 8) | ((n3 & 0xf) << 12);
    new_rows[0] ^= col_down_r0[cv];
    new_rows[1] ^= col_down_r1[cv];
    new_rows[2] ^= col_down_r2[cv];
    new_rows[3] ^= col_down_r3[cv];

    // col 1
    cv = ((n0 >> 4) & 0xf) | (((n1 >> 4) & 0xf) << 4) | (((n2 >> 4) & 0xf) << 8) | (((n3 >> 4) & 0xf) << 12);
    new_rows[0] ^= col_down_r0[cv] << 4;
    new_rows[1] ^= col_down_r1[cv] << 4;
    new_rows[2] ^= col_down_r2[cv] << 4;
    new_rows[3] ^= col_down_r3[cv] << 4;

    // col 2
    cv = ((n0 >> 8) & 0xf) | (((n1 >> 8) & 0xf) << 4) | (((n2 >> 8) & 0xf) << 8) | (((n3 >> 8) & 0xf) << 12);
    new_rows[0] ^= col_down_r0[cv] << 8;
    new_rows[1] ^= col_down_r1[cv] << 8;
    new_rows[2] ^= col_down_r2[cv] << 8;
    new_rows[3] ^= col_down_r3[cv] << 8;

    // col 3
    cv = ((n0 >> 12) & 0xf) | (((n1 >> 12) & 0xf) << 4) | (((n2 >> 12) & 0xf) << 8) | (((n3 >> 12) & 0xf) << 12);
    new_rows[0] ^= col_down_r0[cv] << 12;
    new_rows[1] ^= col_down_r1[cv] << 12;
    new_rows[2] ^= col_down_r2[cv] << 12;
    new_rows[3] ^= col_down_r3[cv] << 12;

    new_rows[0] &= 0xFFFF; new_rows[1] &= 0xFFFF;
    new_rows[2] &= 0xFFFF; new_rows[3] &= 0xFFFF;
    return new_rows;
  }

  /** 左移（C++ move 2）—— 逐行查表 */
  function move_left(rows) {
    const r = rows;
    return [
      r[0] ^ row_left_table[r[0]],
      r[1] ^ row_left_table[r[1]],
      r[2] ^ row_left_table[r[2]],
      r[3] ^ row_left_table[r[3]],
    ];
  }

  /** 右移（C++ move 3）—— 逐行查表 */
  function move_right(rows) {
    const r = rows;
    return [
      r[0] ^ row_right_table[r[0]],
      r[1] ^ row_right_table[r[1]],
      r[2] ^ row_right_table[r[2]],
      r[3] ^ row_right_table[r[3]],
    ];
  }

  const MOVE_FNS = [move_up, move_down, move_left, move_right];

  function execute_move(move, rows) {
    return MOVE_FNS[move](rows);
  }

  // ============================================================
  // 评分函数（纯 Number 运算）
  // ============================================================

  function score_helper(rows, table) {
    return table[rows[0]] + table[rows[1]] + table[rows[2]] + table[rows[3]];
  }

  /**
   * 启发式棋盘评分（行 + 列），完全内联无分配
   * 列评分直接提取每列 nibble 组合后查表，无需创建转置数组
   */
  function score_heur_rows(r) {
    const r0 = r[0], r1 = r[1], r2 = r[2], r3 = r[3];
    return (
      // 行评分
      heur_score_table[r0] + heur_score_table[r1] +
      heur_score_table[r2] + heur_score_table[r3] +
      // 列 0 评分
      heur_score_table[(r0 & 0xf) | ((r1 & 0xf) << 4) | ((r2 & 0xf) << 8) | ((r3 & 0xf) << 12)] +
      // 列 1 评分
      heur_score_table[((r0 >> 4) & 0xf) | (((r1 >> 4) & 0xf) << 4) | (((r2 >> 4) & 0xf) << 8) | (((r3 >> 4) & 0xf) << 12)] +
      // 列 2 评分
      heur_score_table[((r0 >> 8) & 0xf) | (((r1 >> 8) & 0xf) << 4) | (((r2 >> 8) & 0xf) << 8) | (((r3 >> 8) & 0xf) << 12)] +
      // 列 3 评分
      heur_score_table[((r0 >> 12) & 0xf) | (((r1 >> 12) & 0xf) << 4) | (((r2 >> 12) & 0xf) << 8) | (((r3 >> 12) & 0xf) << 12)]
    );
  }

  function score_board_rows(rows) {
    return score_helper(rows, score_table);
  }

  // ============================================================
  // 棋盘统计
  // ============================================================

  function count_empty_rows(rows) {
    return empty_count_table[rows[0]] + empty_count_table[rows[1]] +
      empty_count_table[rows[2]] + empty_count_table[rows[3]];
  }

  function count_distinct_rows(rows) {
    let bitset = 0;
    for (let i = 0; i < 4; i++) {
      let r = rows[i];
      for (let j = 0; j < 4; j++) {
        bitset |= 1 << (r & 0xf);
        r >>= 4;
      }
    }
    bitset >>= 1; // 去掉空格 (bit 0)
    let count = 0;
    while (bitset) {
      bitset &= bitset - 1;
      count++;
    }
    return count;
  }

  // ============================================================
  // Expectimax 搜索
  // ============================================================

  class EvalState {
    constructor() {
      /** @type {Map<string, {depth: number, heuristic: number}>} */
      this.trans_table = new Map();
      this.maxdepth = 0;
      this.curdepth = 0;
      this.cachehits = 0;
      this.moves_evaled = 0;
      this.depth_limit = 0;
    }
  }

  /**
   * 期望层：遍历所有空格，计算 tile 2 和 tile 4 的期望值
   * 使用原地修改 + 恢复，避免大量数组分配
   */
  function score_tilechoose_node(state, rows, cprob) {
    if (cprob < CPROB_THRESH_BASE || state.curdepth >= state.depth_limit) {
      state.maxdepth = Math.max(state.curdepth, state.maxdepth);
      return score_heur_rows(rows);
    }

    // 置换表查询
    if (state.curdepth < CACHE_DEPTH_LIMIT) {
      const key = rows_to_key(rows);
      const entry = state.trans_table.get(key);
      if (entry !== undefined && entry.depth <= state.curdepth) {
        state.cachehits++;
        return entry.heuristic;
      }
    }

    const num_open = count_empty_rows(rows);
    const cprob_per = cprob / num_open;

    let res = 0.0;

    // 遍历 16 个格子，原地修改 rows 避免分配
    // 使用预计算常量数组 TILE2_NIB / TILE4_NIB 替代 1<<(4*c) 计算
    for (let r = 0; r < 4; r++) {
      const orig_row = rows[r];
      let mask = 0xf;
      for (let c = 0; c < 4; c++) {
        if ((orig_row & mask) === 0) {
          // tile 2: nibble = 1
          rows[r] = orig_row | TILE2_NIB[c];
          res += score_move_node(state, rows, cprob_per * 0.9) * 0.9;
          // tile 4: nibble = 2
          rows[r] = orig_row | TILE4_NIB[c];
          res += score_move_node(state, rows, cprob_per * 0.1) * 0.1;
        }
        mask <<= 4;
      }
      rows[r] = orig_row; // 恢复
    }

    res = res / num_open;

    // 存入置换表
    if (state.curdepth < CACHE_DEPTH_LIMIT) {
      const key = rows_to_key(rows);
      state.trans_table.set(key, { depth: state.curdepth, heuristic: res });
    }

    return res;
  }

  /**
   * Max 层：尝试四个方向，取最大评分
   */
  function score_move_node(state, rows, cprob) {
    let best = 0.0;
    state.curdepth++;
    for (let move = 0; move < 4; move++) {
      const new_rows = execute_move(move, rows);
      state.moves_evaled++;
      if (!rows_equal(rows, new_rows)) {
        best = Math.max(best, score_tilechoose_node(state, new_rows, cprob));
      }
    }
    state.curdepth--;
    return best;
  }

  function score_toplevel_move(state, rows, move) {
    const new_rows = execute_move(move, rows);
    if (rows_equal(rows, new_rows)) return 0;
    return score_tilechoose_node(state, new_rows, 1.0) + 1e-6;
  }

  function find_best_move(rows) {
    let best = -Infinity;
    let bestmove = -1;
    const state = new EvalState();
    state.depth_limit = Math.max(3, count_distinct_rows(rows) - 2);

    for (let move = 0; move < 4; move++) {
      const res = score_toplevel_move(state, rows, move);
      if (res > best) {
        best = res;
        bestmove = move;
      }
    }

    return bestmove;
  }

  // ============================================================
  // 游戏 Grid ↔ 行数组 转换
  // ============================================================

  /**
   * 将游戏 Grid 对象转换为 4 行数组
   * Grid 的 cells[col][row]，col=x, row=y
   * 行数组：rows[row] 中 nibble col 存储该格 log2 值
   */
  function grid_to_rows(grid) {
    const rows = [0, 0, 0, 0];
    for (let row = 0; row < 4; row++) {
      let row_val = 0;
      for (let col = 0; col < 4; col++) {
        const tile = grid.cells[col][row];
        const nibble = tile ? Math.round(Math.log2(tile.value)) : 0;
        row_val |= nibble << (4 * col);
      }
      rows[row] = row_val;
    }
    return rows;
  }

  /**
   * 调试：打印行数组
   */
  function print_rows(rows) {
    const out = [];
    for (let i = 0; i < 4; i++) {
      const line = [];
      let r = rows[i];
      for (let j = 0; j < 4; j++) {
        const n = r & 0xf;
        line.push(String(n === 0 ? 0 : 1 << n).padStart(6));
        r >>= 4;
      }
      out.push(line.join(''));
    }
    console.log(out.join('\n'));
  }

  // ============================================================
  // AI 控制器
  // ============================================================

  class AIController {
    /**
     * @param {GameManager} gameManager
     */
    constructor(gameManager) {
      this.gm = gameManager;
      this.running = false;
      this.paused = false;
      this._timeoutId = null;
      this._stepBound = this._step.bind(this);
      this._moveCount = 0;
      this._totalTime = 0;
      this._setupHooks();
    }

    _setupHooks() {
      this.gm.inputManager.on('keepPlaying', () => {
        if (this.paused && this.running) {
          console.log('continue');
          this.paused = false;
          this._scheduleStep();
        }
      });

      this.gm.inputManager.on('restart', () => {
        if (this.running) {
          console.log('stop by reset');
          this.stop();
        }
      });
    }

    start() {
      if (this.running) {
        console.log('algo is on');
        return;
      }
      this._ensureTables();

      this.running = true;
      this.paused = false;
      this._moveCount = 0;
      this._totalTime = 0;
      console.log('start');
      this._scheduleStep();
    }

    stop() {
      this.running = false;
      this.paused = false;
      if (this._timeoutId !== null) {
        clearTimeout(this._timeoutId);
        this._timeoutId = null;
      }
      if (this._moveCount > 0) {
        console.log(
          `stop after ${this._moveCount} steps in ${(this._totalTime / this._moveCount).toFixed(0)} ms on average`
        );
      } else {
        console.log('stop');
      }
    }

    _scheduleStep() {
      // 短延迟让浏览器渲染动画（requestAnimationFrame 之后 + 微延迟）
      this._timeoutId = setTimeout(this._stepBound, 80);
    }

    /**
     * 确保查找表已初始化（供 start 和 stepOnce 复用）
     */
    _ensureTables() {
      if (!tablesInitialized) {
        console.log('init 65536 data');
        const t0 = performance.now();
        init_tables();
        console.log(`init finish in (${(performance.now() - t0).toFixed(0)} ms)`);
      }
    }

    /**
     * 读取当前棋盘、计算最佳移动，并在计算后再次校验棋盘是否被用户修改。
     * 若棋盘已变化则重新计算，最多重试 3 次。
     * 返回 { move, elapsed }，无有效移动时返回 null。
     */
    _computeMove() {
      const MAX_RETRIES = 3;
      let rows = grid_to_rows(this.gm.grid);
      let move, elapsed;
      let retries = 0;

      do {
        const t0 = performance.now();
        move = find_best_move(rows);
        elapsed = performance.now() - t0;

        if (move < 0) return null;

        // 重新读取棋盘，检查计算期间是否被用户手动修改
        const currentRows = grid_to_rows(this.gm.grid);
        if (!rows_equal(rows, currentRows)) {
          rows = currentRows;
          retries++;
          if (retries >= MAX_RETRIES) {
            console.log('stop by user actions');
            return null;
          }
          console.log('recalc by user action');
          continue;
        }
        break;
      } while (true);

      return { move, elapsed };
    }

    /**
     * 单步移动：仅根据当前棋盘执行一次移动，不进入自动循环。
     * 可在控制台直接调用 stepAI()，也可在自动运行期间手动穿插调用。
     */
    stepOnce() {
      this._ensureTables();

      // 检查游戏是否终止
      if (this.gm.isGameTerminated()) {
        if (this.gm.won && !this.gm.keepPlaying) {
          console.log('stop by win');
        } else {
          console.log('stop by finish');
        }
        return;
      }

      const result = this._computeMove();
      if (!result) {
        console.log('no valid move');
        return;
      }

      const { move, elapsed } = result;
      const dirMapping = [0, 2, 3, 1];
      const gameDir = dirMapping[move];
      const moveNames = ['up', 'right', 'down', 'left'];

      console.log(`move ${moveNames[gameDir]} in ${elapsed.toFixed(0)} ms`)
      this.gm.move(gameDir);
    }

    _step() {
      if (!this.running || this.paused) return;

      // 检查游戏是否终止
      if (this.gm.isGameTerminated()) {
        if (this.gm.won && !this.gm.keepPlaying) {
          console.log('pause by win');
          this.paused = true;
          return;
        }
        console.log('stop by finish');
        this.stop();
        return;
      }

      const result = this._computeMove();
      if (!result) {
        console.log('stop by no valid move');
        this.stop();
        return;
      }

      const { move, elapsed } = result;
      this._moveCount++;
      this._totalTime += elapsed;

      // 方向映射：C++ → 游戏
      const dirMapping = [0, 2, 3, 1];
      const gameDir = dirMapping[move];
      const moveNames = ['up', 'right', 'down', 'left'];

      console.log(`step ${this._moveCount}\tin ${elapsed.toFixed(0)}\tms`);
      this.gm.move(gameDir);
      this._scheduleStep();
    }
  }

  // ============================================================
  // 全局 API
  // ============================================================

  function createAI(gameManager) {
    if (window._aiController) {
      window._aiController.stop();
    }
    window._aiController = new AIController(gameManager);

    window.startAI = function () { window._aiController.start(); };
    window.stopAI = function () { window._aiController.stop(); };
    window.stepAI = function () { window._aiController.stepOnce(); };

    const startButton = document.getElementById('algo-start-button');
    const stopButton = document.getElementById('algo-stop-button');
    const hintButton = document.getElementById('algo-hint-button');

    startButton.addEventListener('click', () => startAI());
    stopButton.addEventListener('click', () => stopAI());
    hintButton.addEventListener('click', () => stepAI());

    console.log('algo init finish');
    return window._aiController;
  }

  window.AIController = AIController;
  window.createAI = createAI;
  // 调试用
  window._ai_grid_to_rows = grid_to_rows;
  window._ai_find_best_move = find_best_move;
  window._ai_print_rows = print_rows;

  // ============================================================
  // 自动初始化：检测 window.gameManager 赋值时自动创建 AI 控制器
  // ============================================================

  // 若页面中已有 gameManager（如热重载场景），立即初始化
  if (window.gameManager && !window._aiController) {
    createAI(window.gameManager);
  }

  // 否则拦截 window.gameManager 的赋值，在 application.js 设置时自动触发
  let _gm = window.gameManager || undefined;
  Object.defineProperty(window, 'gameManager', {
    configurable: true,
    enumerable: true,
    get: function () { return _gm; },
    set: function (val) {
      _gm = val;
      if (val && !window._aiController) {
        createAI(val);
      }
    }
  });
})();
