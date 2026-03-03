const EPSILON = 1e-12;
const DOT_ALIASES = ["。", "．", "｡", "﹒"];
const MINUS_ALIASES = ["－", "﹣", "−"];
const PERIOD_CODES = new Set(["Period", "NumpadDecimal"]);
const PERIOD_KEYS = new Set([".", "。", "．", "｡", "﹒"]);
const MINUS_KEYS = new Set(["-", "－", "﹣", "−"]);

const state = {
  distributions: [],
  opponents: [],
  currentOpponent: null,
  enemySelection: null,
  currentModel: null,
  playerSelection: null,
  battleRound: 0,
  fixedStats: {
    total: 0,
    playerWins: 0,
    enemyWins: 0,
    ties: 0,
  },
};

const ui = {
  enemyStatus: document.querySelector("#enemy-status"),
  playerStatus: document.querySelector("#player-status"),
  fixedStats: document.querySelector("#fixed-stats"),
  enemySelect: document.querySelector("#enemy-select"),
  modelSelect: document.querySelector("#model-select"),
  paramForm: document.querySelector("#param-form"),
  distributionHelp: document.querySelector("#distribution-help"),
  battleLog: document.querySelector("#battle-log"),
  lockEnemyBtn: document.querySelector("#lock-enemy-btn"),
  lockModelBtn: document.querySelector("#lock-model-btn"),
  lockParamsBtn: document.querySelector("#lock-params-btn"),
  duelBtn: document.querySelector("#duel-btn"),
  backParamBtn: document.querySelector("#back-param-btn"),
  backModelBtn: document.querySelector("#back-model-btn"),
  backEnemyBtn: document.querySelector("#back-enemy-btn"),
  clearLogBtn: document.querySelector("#clear-log-btn"),
};

function normalizeNumericInput(text) {
  let normalized = String(text);
  for (const dot of DOT_ALIASES) normalized = normalized.replaceAll(dot, ".");
  for (const minus of MINUS_ALIASES) normalized = normalized.replaceAll(minus, "-");
  return normalized;
}

function formatNumber(value) {
  if (Math.abs(value - Math.round(value)) < EPSILON) {
    return String(Math.round(value));
  }
  return value.toFixed(4);
}

function parseFloatStrict(raw) {
  const text = normalizeNumericInput(raw).trim();
  if (!text) throw new Error("请输入数字");
  const value = Number.parseFloat(text);
  if (Number.isNaN(value)) throw new Error("请输入有效数字");
  return value;
}

function parseIntStrict(raw) {
  const text = normalizeNumericInput(raw).trim();
  if (!/^-?\d+$/.test(text)) throw new Error("请输入整数");
  return Number.parseInt(text, 10);
}

function validatePositive(v) {
  return v > 0 ? null : "必须大于 0";
}

function validateNonPositive(v) {
  return v <= 0 ? null : "必须小于等于 0";
}

function validateNonNegativeInt(v) {
  return Number.isInteger(v) && v >= 0 ? null : "必须是大于等于 0 的整数";
}

function validateZeroToOne(v) {
  return v >= 0 && v <= 1 ? null : "必须在 [0, 1] 区间";
}

function randUniform(a, b) {
  return a + Math.random() * (b - a);
}

function randNormal(mu, sigma) {
  let u1 = 0;
  let u2 = 0;
  while (u1 === 0) u1 = Math.random();
  while (u2 === 0) u2 = Math.random();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mu + sigma * z;
}

function randExponential(lambda) {
  let u = 0;
  while (u === 0) u = Math.random();
  return -Math.log(u) / lambda;
}

function samplePoisson(lam) {
  if (lam < 30) {
    const threshold = Math.exp(-lam);
    let k = 0;
    let product = 1;
    while (product > threshold) {
      k += 1;
      product *= Math.random();
    }
    return k - 1;
  }
  return Math.max(0, Math.round(randNormal(lam, Math.sqrt(lam))));
}

function sampleBinomial(n, p) {
  let count = 0;
  for (let i = 0; i < n; i += 1) {
    if (Math.random() < p) count += 1;
  }
  return count;
}

function sampleGamma(shape, scale) {
  if (shape < 1) {
    const g = sampleGamma(shape + 1, 1);
    let u = 0;
    while (u === 0) u = Math.random();
    return scale * g * Math.pow(u, 1 / shape);
  }

  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    const x = randNormal(0, 1);
    const v0 = 1 + c * x;
    if (v0 <= 0) continue;
    const v = v0 * v0 * v0;
    const u = Math.random();
    if (u < 1 - 0.0331 * Math.pow(x, 4)) return scale * d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return scale * d * v;
  }
}

function sampleBeta(alpha, beta) {
  const gx = sampleGamma(alpha, 1);
  const gy = sampleGamma(beta, 1);
  return gx / (gx + gy);
}

function buildDistributions() {
  return [
    {
      key: "uniform",
      name: "均匀分布 Uniform(a, -a)",
      description: "只需选择下界 a<=0，上界自动为 -a",
      params: [
        { name: "a", label: "a (下界, 要求 <=0)", parser: parseFloatStrict, validator: validateNonPositive },
      ],
      sample: (p) => randUniform(p.a, -p.a),
      expectation: () => 0,
      centerNote: "E[X]=0，按 X 判定。",
    },
    {
      key: "normal",
      name: "正态分布 Normal(0, sigma)",
      description: "均值固定为 0，只需选择标准差 sigma",
      params: [
        { name: "sigma", label: "sigma (标准差)", parser: parseFloatStrict, validator: validatePositive },
      ],
      sample: (p) => randNormal(0, p.sigma),
      expectation: () => 0,
      centerNote: "E[X]=0，按 X 判定。",
    },
    {
      key: "exponential",
      name: "指数分布 Exp(lambda)",
      description: "选择 lambda>0，系统自动中心化",
      params: [{ name: "lambda", label: "lambda", parser: parseFloatStrict, validator: validatePositive }],
      sample: (p) => randExponential(p.lambda),
      expectation: (p) => 1 / p.lambda,
      centerNote: "按 X-1/lambda 判定，期望为 0。",
    },
    {
      key: "poisson",
      name: "泊松分布 Poisson(lam)",
      description: "选择 lam>0，系统自动中心化",
      params: [{ name: "lam", label: "lam", parser: parseFloatStrict, validator: validatePositive }],
      sample: (p) => samplePoisson(p.lam),
      expectation: (p) => p.lam,
      centerNote: "按 X-lam 判定，期望为 0。",
    },
    {
      key: "binomial",
      name: "二项分布 Binomial(n, p)",
      description: "选择 n 与 p，系统自动中心化",
      params: [
        { name: "n", label: "n (非负整数)", parser: parseIntStrict, validator: validateNonNegativeInt },
        { name: "p", label: "p (成功概率)", parser: parseFloatStrict, validator: validateZeroToOne },
      ],
      sample: (p) => sampleBinomial(p.n, p.p),
      expectation: (p) => p.n * p.p,
      centerNote: "按 X-np 判定，期望为 0。",
    },
    {
      key: "gamma",
      name: "伽马分布 Gamma(k, theta)",
      description: "选择 k 与 theta，系统自动中心化",
      params: [
        { name: "k", label: "k (shape)", parser: parseFloatStrict, validator: validatePositive },
        { name: "theta", label: "theta (scale)", parser: parseFloatStrict, validator: validatePositive },
      ],
      sample: (p) => sampleGamma(p.k, p.theta),
      expectation: (p) => p.k * p.theta,
      centerNote: "按 X-k*theta 判定，期望为 0。",
    },
    {
      key: "beta",
      name: "贝塔分布 Beta(alpha, beta)",
      description: "选择 alpha 与 beta，系统自动中心化",
      params: [
        { name: "alpha", label: "alpha", parser: parseFloatStrict, validator: validatePositive },
        { name: "beta", label: "beta", parser: parseFloatStrict, validator: validatePositive },
      ],
      sample: (p) => sampleBeta(p.alpha, p.beta),
      expectation: (p) => p.alpha / (p.alpha + p.beta),
      centerNote: "按 X-alpha/(alpha+beta) 判定，期望为 0。",
    },
  ];
}

function randomParamsFor(modelKey) {
  switch (modelKey) {
    case "uniform":
      return { a: randUniform(-3, 0) };
    case "normal":
      return { sigma: randUniform(0.4, 2.5) };
    case "exponential":
      return { lambda: randUniform(0.2, 1.8) };
    case "poisson":
      return { lam: randUniform(1, 12) };
    case "binomial":
      return { n: Math.floor(randUniform(3, 41)), p: randUniform(0.2, 0.85) };
    case "gamma":
      return { k: randUniform(0.7, 5), theta: randUniform(0.4, 2.8) };
    case "beta":
      return { alpha: randUniform(0.6, 4.5), beta: randUniform(0.6, 4.5) };
    default:
      throw new Error(`未知分布: ${modelKey}`);
  }
}

function buildOpponents(distributions) {
  const byKey = Object.fromEntries(distributions.map((d) => [d.key, d]));
  return [
    {
      name: "均值工程师",
      style: "偏稳健：固定正态分布，零均值下控制波动",
      strategy: () => ({ model: byKey.normal, params: { sigma: 0.8 } }),
    },
    {
      name: "爆发投机者",
      style: "高波动：固定伽马分布，追求偏态爆发",
      strategy: () => ({ model: byKey.gamma, params: { k: 1.1, theta: 3.4 } }),
    },
    {
      name: "离散操盘手",
      style: "离散流派：固定二项分布，收益相对可控",
      strategy: () => ({ model: byKey.binomial, params: { n: 22, p: 0.56 } }),
    },
    {
      name: "泊松计数师",
      style: "计数流派：固定泊松分布，重视出现频次",
      strategy: () => ({ model: byKey.poisson, params: { lam: 6.8 } }),
    },
    {
      name: "全随机模拟器",
      style: "不可预测：随机分布 + 随机参数",
      strategy: () => {
        const model = distributions[Math.floor(Math.random() * distributions.length)];
        return { model, params: randomParamsFor(model.key) };
      },
    },
  ];
}

function effectiveParams(selection) {
  if (!selection) return {};
  if (selection.model.key === "uniform") {
    return { a: selection.params.a, b: -selection.params.a };
  }
  if (selection.model.key === "normal") {
    return { mu: 0, sigma: selection.params.sigma };
  }
  return { ...selection.params };
}

function formatParams(params) {
  return Object.entries(params)
    .map(([k, v]) => `${k}=${Number.isFinite(v) ? formatNumber(v) : String(v)}`)
    .join(", ");
}

function formatSelectionParams(selection) {
  const chosen = formatParams(selection.params);
  const actual = formatParams(effectiveParams(selection));
  return chosen === actual ? chosen : `${chosen}; 推导后 ${actual}`;
}

function runDuel(playerSelection, enemySelection) {
  const playerRaw = playerSelection.model.sample(playerSelection.params);
  const enemyRaw = enemySelection.model.sample(enemySelection.params);
  const playerMean = playerSelection.model.expectation(playerSelection.params);
  const enemyMean = enemySelection.model.expectation(enemySelection.params);
  const playerScore = playerRaw - playerMean;
  const enemyScore = enemyRaw - enemyMean;

  let winnerText = "平局";
  if (playerScore > enemyScore + EPSILON) winnerText = "你获胜";
  else if (enemyScore > playerScore + EPSILON) winnerText = "电脑获胜";

  return { playerRaw, enemyRaw, playerMean, enemyMean, playerScore, enemyScore, winnerText };
}

function appendLog(line = "") {
  ui.battleLog.textContent += `${line}\n`;
  ui.battleLog.scrollTop = ui.battleLog.scrollHeight;
}

function updateEnemyStatus() {
  if (!state.enemySelection) {
    ui.enemyStatus.textContent = "未锁定";
    return;
  }
  ui.enemyStatus.textContent = `${state.enemySelection.model.name} (${formatSelectionParams(state.enemySelection)})`;
}

function updatePlayerStatus() {
  if (state.playerSelection) {
    ui.playerStatus.textContent = `${state.playerSelection.model.name} (${formatSelectionParams(
      state.playerSelection
    )})`;
    return;
  }
  if (state.currentModel) {
    ui.playerStatus.textContent = `模型已锁定: ${state.currentModel.name}，参数未锁定`;
    return;
  }
  ui.playerStatus.textContent = "未锁定";
}

function updateFixedStatsStatus() {
  const { total, playerWins, enemyWins, ties } = state.fixedStats;
  if (total === 0) {
    ui.fixedStats.textContent = "未开始判定";
    return;
  }
  const winRate = (100 * playerWins) / total;
  ui.fixedStats.textContent = `${playerWins}胜 ${enemyWins}负 ${ties}平，胜率=${winRate.toFixed(2)}%（${playerWins}/${total}）`;
}

function updateDuelButtonText() {
  if (state.fixedStats.total > 0) {
    ui.duelBtn.textContent = "立刻再次判定（保持敌人、模型、参数不变）";
  } else {
    ui.duelBtn.textContent = "开始判定";
  }
}

function resetFixedStats() {
  state.battleRound = 0;
  state.fixedStats = { total: 0, playerWins: 0, enemyWins: 0, ties: 0 };
  updateFixedStatsStatus();
  updateDuelButtonText();
}

function setControlStates({
  modelEnabled = false,
  lockModelEnabled = false,
  lockParamsEnabled = false,
  duelEnabled = false,
  backParamEnabled = false,
  backModelEnabled = false,
  backEnemyEnabled = false,
} = {}) {
  ui.modelSelect.disabled = !modelEnabled;
  ui.lockModelBtn.disabled = !lockModelEnabled;
  ui.lockParamsBtn.disabled = !lockParamsEnabled;
  ui.duelBtn.disabled = !duelEnabled;
  ui.backParamBtn.disabled = !backParamEnabled;
  ui.backModelBtn.disabled = !backModelEnabled;
  ui.backEnemyBtn.disabled = !backEnemyEnabled;
}

function clearParamForm() {
  ui.paramForm.innerHTML = "";
}

function injectAtCursor(input, text) {
  const start = input.selectionStart ?? input.value.length;
  const end = input.selectionEnd ?? input.value.length;
  input.value = `${input.value.slice(0, start)}${text}${input.value.slice(end)}`;
  const caret = start + text.length;
  input.setSelectionRange(caret, caret);
  input.dispatchEvent(new Event("input", { bubbles: true }));
}

function attachNumericInputBehavior(input) {
  input.addEventListener("keydown", (event) => {
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    if (PERIOD_CODES.has(event.code) || PERIOD_KEYS.has(event.key)) {
      event.preventDefault();
      injectAtCursor(input, ".");
      return;
    }
    if (MINUS_KEYS.has(event.key)) {
      event.preventDefault();
      injectAtCursor(input, "-");
    }
  });

  input.addEventListener("input", () => {
    const raw = input.value;
    const normalized = normalizeNumericInput(raw);
    if (raw !== normalized) {
      const pos = input.selectionStart ?? normalized.length;
      input.value = normalized;
      input.setSelectionRange(pos, pos);
    }
  });
}

function buildParamForm(model, preset = null) {
  clearParamForm();
  model.params.forEach((spec) => {
    const row = document.createElement("div");
    row.className = "param-row";

    const label = document.createElement("label");
    label.htmlFor = `param-${spec.name}`;
    label.textContent = spec.label;

    const input = document.createElement("input");
    input.id = `param-${spec.name}`;
    input.name = spec.name;
    input.autocomplete = "off";
    input.inputMode = "decimal";
    input.placeholder = "请输入参数";
    input.value = preset && spec.name in preset ? String(preset[spec.name]) : "";
    attachNumericInputBehavior(input);

    row.append(label, input);
    ui.paramForm.append(row);
  });
}

function readParams(model) {
  const values = {};
  for (const spec of model.params) {
    const input = ui.paramForm.querySelector(`[name='${spec.name}']`);
    const raw = input ? input.value : "";
    let parsed;
    try {
      parsed = spec.parser(raw);
    } catch (err) {
      alert(`${spec.name} 输入格式错误：${err.message}`);
      return null;
    }
    const errMsg = spec.validator(parsed);
    if (errMsg) {
      alert(`${spec.name} 无效：${errMsg}`);
      return null;
    }
    values[spec.name] = parsed;
  }
  return values;
}

function renderDistributionHelp() {
  const lines = [];
  state.distributions.forEach((dist, i) => {
    lines.push(`${i + 1}. ${dist.name}`);
    lines.push(`   ${dist.description}`);
    lines.push(`   ${dist.centerNote}`);
  });
  ui.distributionHelp.textContent = lines.join("\n");
}

function resetToEnemyLevel() {
  state.currentOpponent = null;
  state.enemySelection = null;
  state.currentModel = null;
  state.playerSelection = null;
  resetFixedStats();
  clearParamForm();
  updateEnemyStatus();
  updatePlayerStatus();
  setControlStates({
    modelEnabled: false,
    lockModelEnabled: false,
    lockParamsEnabled: false,
    duelEnabled: false,
    backParamEnabled: false,
    backModelEnabled: false,
    backEnemyEnabled: false,
  });
}

function onLockEnemy() {
  const idx = ui.enemySelect.selectedIndex;
  if (idx < 0) return;
  state.currentOpponent = state.opponents[idx];
  state.enemySelection = state.currentOpponent.strategy();
  state.currentModel = null;
  state.playerSelection = null;
  resetFixedStats();
  clearParamForm();
  updateEnemyStatus();
  updatePlayerStatus();
  setControlStates({
    modelEnabled: true,
    lockModelEnabled: true,
    lockParamsEnabled: false,
    duelEnabled: false,
    backParamEnabled: false,
    backModelEnabled: false,
    backEnemyEnabled: true,
  });
  appendLog(`[系统] 敌人已锁定：${state.currentOpponent.name}`);
}

function onLockModel() {
  const idx = ui.modelSelect.selectedIndex;
  if (idx < 0 || !state.enemySelection) return;
  state.currentModel = state.distributions[idx];
  state.playerSelection = null;
  resetFixedStats();
  buildParamForm(state.currentModel);
  updatePlayerStatus();
  setControlStates({
    modelEnabled: true,
    lockModelEnabled: true,
    lockParamsEnabled: true,
    duelEnabled: false,
    backParamEnabled: false,
    backModelEnabled: true,
    backEnemyEnabled: true,
  });
  appendLog(`[系统] 模型已锁定：${state.currentModel.name}`);
}

function onLockParams() {
  if (!state.currentModel) return;
  const parsed = readParams(state.currentModel);
  if (!parsed) return;
  state.playerSelection = { model: state.currentModel, params: parsed };
  resetFixedStats();
  updatePlayerStatus();
  setControlStates({
    modelEnabled: true,
    lockModelEnabled: true,
    lockParamsEnabled: true,
    duelEnabled: true,
    backParamEnabled: true,
    backModelEnabled: true,
    backEnemyEnabled: true,
  });
  appendLog("[系统] 参数已锁定，可开始判定。");
}

function onDuel() {
  if (!state.playerSelection || !state.enemySelection) return;
  const isInstantRematch = state.fixedStats.total > 0;
  state.battleRound += 1;
  const result = runDuel(state.playerSelection, state.enemySelection);

  appendLog("");
  appendLog(`=== 第 ${state.battleRound} 局 ===`);
  appendLog(`你：${state.playerSelection.model.name} (${formatSelectionParams(state.playerSelection)})`);
  appendLog(`电脑：${state.enemySelection.model.name} (${formatSelectionParams(state.enemySelection)})`);
  appendLog(
    `你：原始采样=${formatNumber(result.playerRaw)}, 理论期望=${formatNumber(
      result.playerMean
    )}, 判定值=${formatNumber(result.playerScore)}`
  );
  appendLog(
    `电脑：原始采样=${formatNumber(result.enemyRaw)}, 理论期望=${formatNumber(
      result.enemyMean
    )}, 判定值=${formatNumber(result.enemyScore)}`
  );
  appendLog(`结果：${result.winnerText}`);

  state.fixedStats.total += 1;
  if (result.winnerText === "你获胜") state.fixedStats.playerWins += 1;
  else if (result.winnerText === "电脑获胜") state.fixedStats.enemyWins += 1;
  else state.fixedStats.ties += 1;

  updateFixedStatsStatus();
  updateDuelButtonText();

  if (isInstantRematch) {
    const { playerWins, enemyWins, ties, total } = state.fixedStats;
    const rate = ((100 * playerWins) / total).toFixed(2);
    appendLog(`[固定配置统计] ${playerWins}胜 ${enemyWins}负 ${ties}平，玩家胜率=${rate}%（${playerWins}/${total}）`);
  }
}

function onBackToParams() {
  if (!state.currentModel) return;
  const preset = state.playerSelection ? state.playerSelection.params : null;
  state.playerSelection = null;
  resetFixedStats();
  buildParamForm(state.currentModel, preset);
  updatePlayerStatus();
  setControlStates({
    modelEnabled: true,
    lockModelEnabled: true,
    lockParamsEnabled: true,
    duelEnabled: false,
    backParamEnabled: false,
    backModelEnabled: true,
    backEnemyEnabled: true,
  });
  appendLog("[系统] 已返回参数选择。敌人和模型保持不变。");
}

function onBackToModel() {
  if (!state.enemySelection) return;
  state.currentModel = null;
  state.playerSelection = null;
  resetFixedStats();
  clearParamForm();
  updatePlayerStatus();
  setControlStates({
    modelEnabled: true,
    lockModelEnabled: true,
    lockParamsEnabled: false,
    duelEnabled: false,
    backParamEnabled: false,
    backModelEnabled: false,
    backEnemyEnabled: true,
  });
  appendLog("[系统] 已返回模型选择。敌人保持不变。");
}

function onBackToEnemy() {
  resetToEnemyLevel();
  appendLog("[系统] 已返回敌人选择。模型和参数已重置。");
}

function setupSelectors() {
  ui.enemySelect.innerHTML = "";
  state.opponents.forEach((op) => {
    const opt = document.createElement("option");
    opt.textContent = `${op.name} - ${op.style}`;
    ui.enemySelect.append(opt);
  });

  ui.modelSelect.innerHTML = "";
  state.distributions.forEach((dist) => {
    const opt = document.createElement("option");
    opt.textContent = `${dist.name} - ${dist.description}`;
    ui.modelSelect.append(opt);
  });
}

function bindEvents() {
  ui.lockEnemyBtn.addEventListener("click", onLockEnemy);
  ui.lockModelBtn.addEventListener("click", onLockModel);
  ui.lockParamsBtn.addEventListener("click", onLockParams);
  ui.duelBtn.addEventListener("click", onDuel);
  ui.backParamBtn.addEventListener("click", onBackToParams);
  ui.backModelBtn.addEventListener("click", onBackToModel);
  ui.backEnemyBtn.addEventListener("click", onBackToEnemy);
  ui.clearLogBtn.addEventListener("click", () => {
    ui.battleLog.textContent = "";
  });
}

function init() {
  state.distributions = buildDistributions();
  state.opponents = buildOpponents(state.distributions);
  setupSelectors();
  renderDistributionHelp();
  bindEvents();
  resetToEnemyLevel();
  appendLog("[系统] Web 版已加载，先锁定敌人，再开始对战。");
}

init();
