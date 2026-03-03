from __future__ import annotations

import math
import random
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import Any, Callable


EPSILON = 1e-12
DOT_ALIASES = ("。", "．", "｡", "﹒")
MINUS_ALIASES = ("－", "﹣", "−")
PERIOD_KEYSYMS = {"period", "KP_Decimal", "KP_Separator", "decimal"}
PERIOD_KEYCODES = {190, 110}


@dataclass(frozen=True)
class ParamSpec:
    name: str
    prompt: str
    caster: Callable[[str], Any]
    validator: Callable[[Any], str | None]


@dataclass(frozen=True)
class DistributionSpec:
    key: str
    name: str
    description: str
    params: tuple[ParamSpec, ...]
    sampler: Callable[[random.Random, dict[str, Any]], float]
    expectation: Callable[[dict[str, Any]], float]
    joint_validator: Callable[[dict[str, Any]], str | None]
    centering_note: str


@dataclass(frozen=True)
class Selection:
    model: DistributionSpec
    params: dict[str, Any]


@dataclass(frozen=True)
class Opponent:
    name: str
    style: str
    strategy: Callable[[random.Random], Selection]


@dataclass(frozen=True)
class DuelResult:
    player_raw: float
    enemy_raw: float
    player_mean: float
    enemy_mean: float
    player_score: float
    enemy_score: float
    winner_text: str


def normalize_numeric_input(text: str) -> str:
    normalized = text
    for dot in DOT_ALIASES:
        normalized = normalized.replace(dot, ".")
    for minus in MINUS_ALIASES:
        normalized = normalized.replace(minus, "-")
    return normalized


def int_parser(text: str) -> int:
    return int(normalize_numeric_input(text).strip())


def float_parser(text: str) -> float:
    return float(normalize_numeric_input(text).strip())


def positive(value: float) -> str | None:
    if value > 0:
        return None
    return "必须大于 0"


def non_positive(value: float) -> str | None:
    if value <= 0:
        return None
    return "必须小于等于 0"


def non_negative_int(value: int) -> str | None:
    if value >= 0:
        return None
    return "必须是大于等于 0 的整数"


def zero_to_one(value: float) -> str | None:
    if 0 <= value <= 1:
        return None
    return "必须在 [0, 1] 区间"


def always_valid(_: dict[str, Any]) -> str | None:
    return None


def sample_poisson(rng: random.Random, lam: float) -> int:
    if lam < 30:
        threshold = math.exp(-lam)
        k = 0
        product = 1.0
        while product > threshold:
            k += 1
            product *= rng.random()
        return k - 1

    value = int(round(rng.gauss(lam, math.sqrt(lam))))
    return max(0, value)


def format_number(value: float) -> str:
    if abs(value - round(value)) < EPSILON:
        return str(int(round(value)))
    return f"{value:.4f}"


def build_distributions() -> list[DistributionSpec]:
    return [
        DistributionSpec(
            key="uniform",
            name="均匀分布 Uniform(a, -a)",
            description="只需选择下界 a<=0，上界自动为 -a",
            params=(
                ParamSpec("a", "请输入 a (下界, 要求 <=0): ", float_parser, non_positive),
            ),
            sampler=lambda rng, p: rng.uniform(p["a"], -p["a"]),
            expectation=lambda p: 0.0,
            joint_validator=always_valid,
            centering_note="E[X]=0，按 X 判定。",
        ),
        DistributionSpec(
            key="normal",
            name="正态分布 Normal(0, sigma)",
            description="均值固定为 0，只需选择标准差 sigma",
            params=(
                ParamSpec("sigma", "请输入 sigma (标准差): ", float_parser, positive),
            ),
            sampler=lambda rng, p: rng.gauss(0.0, p["sigma"]),
            expectation=lambda p: 0.0,
            joint_validator=always_valid,
            centering_note="E[X]=0，按 X 判定。",
        ),
        DistributionSpec(
            key="exponential",
            name="指数分布 Exp(lambda)",
            description="选择 lambda>0，系统自动中心化",
            params=(
                ParamSpec("lambda", "请输入 lambda: ", float_parser, positive),
            ),
            sampler=lambda rng, p: rng.expovariate(p["lambda"]),
            expectation=lambda p: 1.0 / p["lambda"],
            joint_validator=always_valid,
            centering_note="按 X-1/lambda 判定，期望为 0。",
        ),
        DistributionSpec(
            key="poisson",
            name="泊松分布 Poisson(lam)",
            description="选择 lam>0，系统自动中心化",
            params=(
                ParamSpec("lam", "请输入 lam: ", float_parser, positive),
            ),
            sampler=lambda rng, p: float(sample_poisson(rng, p["lam"])),
            expectation=lambda p: p["lam"],
            joint_validator=always_valid,
            centering_note="按 X-lam 判定，期望为 0。",
        ),
        DistributionSpec(
            key="binomial",
            name="二项分布 Binomial(n, p)",
            description="选择 n 与 p，系统自动中心化",
            params=(
                ParamSpec("n", "请输入 n (非负整数): ", int_parser, non_negative_int),
                ParamSpec("p", "请输入 p (成功概率): ", float_parser, zero_to_one),
            ),
            sampler=lambda rng, p: float(sum(1 for _ in range(p["n"]) if rng.random() < p["p"])),
            expectation=lambda p: float(p["n"] * p["p"]),
            joint_validator=always_valid,
            centering_note="按 X-np 判定，期望为 0。",
        ),
        DistributionSpec(
            key="gamma",
            name="伽马分布 Gamma(k, theta)",
            description="选择 k 与 theta，系统自动中心化",
            params=(
                ParamSpec("k", "请输入 k (shape): ", float_parser, positive),
                ParamSpec("theta", "请输入 theta (scale): ", float_parser, positive),
            ),
            sampler=lambda rng, p: rng.gammavariate(p["k"], p["theta"]),
            expectation=lambda p: p["k"] * p["theta"],
            joint_validator=always_valid,
            centering_note="按 X-k*theta 判定，期望为 0。",
        ),
        DistributionSpec(
            key="beta",
            name="贝塔分布 Beta(alpha, beta)",
            description="选择 alpha 与 beta，系统自动中心化",
            params=(
                ParamSpec("alpha", "请输入 alpha: ", float_parser, positive),
                ParamSpec("beta", "请输入 beta: ", float_parser, positive),
            ),
            sampler=lambda rng, p: rng.betavariate(p["alpha"], p["beta"]),
            expectation=lambda p: p["alpha"] / (p["alpha"] + p["beta"]),
            joint_validator=always_valid,
            centering_note="按 X-alpha/(alpha+beta) 判定，期望为 0。",
        ),
    ]


def effective_params(model: DistributionSpec, params: dict[str, Any]) -> dict[str, Any]:
    if model.key == "uniform":
        return {"a": params["a"], "b": -params["a"]}
    if model.key == "normal":
        return {"mu": 0.0, "sigma": params["sigma"]}
    return dict(params)


def format_params(params: dict[str, Any]) -> str:
    chunks: list[str] = []
    for key, value in params.items():
        if isinstance(value, float):
            chunks.append(f"{key}={format_number(value)}")
        else:
            chunks.append(f"{key}={value}")
    return ", ".join(chunks)


def format_selection_params(selection: Selection) -> str:
    chosen_text = format_params(selection.params)
    actual_text = format_params(effective_params(selection.model, selection.params))
    if chosen_text == actual_text:
        return chosen_text
    return f"{chosen_text}; 推导后 {actual_text}"


def random_params_for(model_key: str, rng: random.Random) -> dict[str, Any]:
    if model_key == "uniform":
        return {"a": rng.uniform(-3, 0)}
    if model_key == "normal":
        return {"sigma": rng.uniform(0.4, 2.5)}
    if model_key == "exponential":
        return {"lambda": rng.uniform(0.2, 1.8)}
    if model_key == "poisson":
        return {"lam": rng.uniform(1.0, 12.0)}
    if model_key == "binomial":
        return {"n": rng.randint(3, 40), "p": rng.uniform(0.2, 0.85)}
    if model_key == "gamma":
        return {"k": rng.uniform(0.7, 5.0), "theta": rng.uniform(0.4, 2.8)}
    if model_key == "beta":
        return {"alpha": rng.uniform(0.6, 4.5), "beta": rng.uniform(0.6, 4.5)}
    raise ValueError(f"未知分布: {model_key}")


def build_opponents(distributions: list[DistributionSpec]) -> list[Opponent]:
    by_key = {item.key: item for item in distributions}

    return [
        Opponent(
            name="均值工程师",
            style="偏稳健：固定正态分布，零均值下控制波动",
            strategy=lambda _rng: Selection(by_key["normal"], {"sigma": 0.8}),
        ),
        Opponent(
            name="爆发投机者",
            style="高波动：固定伽马分布，追求偏态爆发",
            strategy=lambda _rng: Selection(by_key["gamma"], {"k": 1.1, "theta": 3.4}),
        ),
        Opponent(
            name="离散操盘手",
            style="离散流派：固定二项分布，收益相对可控",
            strategy=lambda _rng: Selection(by_key["binomial"], {"n": 22, "p": 0.56}),
        ),
        Opponent(
            name="泊松计数师",
            style="计数流派：固定泊松分布，重视出现频次",
            strategy=lambda _rng: Selection(by_key["poisson"], {"lam": 6.8}),
        ),
        Opponent(
            name="全随机模拟器",
            style="不可预测：随机分布 + 随机参数",
            strategy=lambda rng: (
                lambda model: Selection(model, random_params_for(model.key, rng))
            )(rng.choice(distributions)),
        ),
    ]


def run_duel(rng: random.Random, player: Selection, enemy: Selection) -> DuelResult:
    player_raw = player.model.sampler(rng, player.params)
    enemy_raw = enemy.model.sampler(rng, enemy.params)
    player_mean = player.model.expectation(player.params)
    enemy_mean = enemy.model.expectation(enemy.params)
    player_score = player_raw - player_mean
    enemy_score = enemy_raw - enemy_mean

    if player_score > enemy_score + EPSILON:
        winner_text = "你获胜"
    elif enemy_score > player_score + EPSILON:
        winner_text = "电脑获胜"
    else:
        winner_text = "平局"

    return DuelResult(
        player_raw=player_raw,
        enemy_raw=enemy_raw,
        player_mean=player_mean,
        enemy_mean=enemy_mean,
        player_score=player_score,
        enemy_score=enemy_score,
        winner_text=winner_text,
    )


class ProbabilityBattleApp:
    def __init__(self) -> None:
        self.rng = random.Random()
        self.distributions = build_distributions()
        self.opponents = build_opponents(self.distributions)

        self.current_opponent: Opponent | None = None
        self.enemy_selection: Selection | None = None
        self.current_model: DistributionSpec | None = None
        self.player_selection: Selection | None = None
        self.param_vars: dict[str, tk.StringVar] = {}
        self.battle_round = 0
        self.fixed_total = 0
        self.fixed_player_wins = 0
        self.fixed_enemy_wins = 0
        self.fixed_ties = 0

        self.root = tk.Tk()
        self.root.title("概率对战 Demo（可视化版）")
        self.root.geometry("980x680")
        self.root.minsize(900, 620)

        self.enemy_status_var = tk.StringVar(value="未锁定")
        self.player_status_var = tk.StringVar(value="未锁定")
        self.fixed_stats_var = tk.StringVar(value="未开始判定")

        self._build_ui()
        self._reset_to_opponent_level()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

        header = ttk.Frame(self.root, padding=(12, 10))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(
            header,
            text="规则：判定值 = 采样值 - 理论期望。双方判定值的数学期望都为 0，判定值更大者胜。",
            justify="left",
        ).grid(row=0, column=0, sticky="w")

        status = ttk.LabelFrame(self.root, text="当前锁定配置", padding=(12, 8))
        status.grid(row=1, column=0, padx=12, pady=(0, 8), sticky="ew")
        status.columnconfigure(1, weight=1)
        ttk.Label(status, text="敌人：").grid(row=0, column=0, sticky="nw", padx=(0, 6))
        ttk.Label(status, textvariable=self.enemy_status_var, justify="left").grid(
            row=0, column=1, sticky="w"
        )
        ttk.Label(status, text="你：").grid(row=1, column=0, sticky="nw", padx=(0, 6))
        ttk.Label(status, textvariable=self.player_status_var, justify="left").grid(
            row=1, column=1, sticky="w"
        )
        ttk.Label(status, text="固定配置统计：").grid(row=2, column=0, sticky="nw", padx=(0, 6))
        ttk.Label(status, textvariable=self.fixed_stats_var, justify="left").grid(
            row=2, column=1, sticky="w"
        )

        main = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        main.grid(row=2, column=0, sticky="nsew")
        main.columnconfigure(0, weight=0)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(1, weight=1)

        control_panel = ttk.LabelFrame(main, text="操作区", padding=10)
        control_panel.grid(row=0, column=0, rowspan=2, sticky="nsw")

        row = 0
        ttk.Label(control_panel, text="1) 选择敌人").grid(row=row, column=0, sticky="w")
        row += 1
        self.opponent_combo = ttk.Combobox(
            control_panel,
            state="readonly",
            width=34,
            values=[f"{op.name} - {op.style}" for op in self.opponents],
        )
        self.opponent_combo.grid(row=row, column=0, pady=(2, 6), sticky="ew")
        row += 1
        self.lock_opponent_btn = ttk.Button(
            control_panel, text="锁定敌人", command=self.lock_opponent
        )
        self.lock_opponent_btn.grid(row=row, column=0, pady=(0, 10), sticky="ew")

        row += 1
        ttk.Label(control_panel, text="2) 选择模型").grid(row=row, column=0, sticky="w")
        row += 1
        self.model_combo = ttk.Combobox(
            control_panel,
            state="disabled",
            width=34,
            values=[f"{d.name} - {d.description}" for d in self.distributions],
        )
        self.model_combo.grid(row=row, column=0, pady=(2, 6), sticky="ew")
        row += 1
        self.lock_model_btn = ttk.Button(
            control_panel, text="锁定模型", command=self.lock_model, state="disabled"
        )
        self.lock_model_btn.grid(row=row, column=0, pady=(0, 10), sticky="ew")

        row += 1
        ttk.Label(control_panel, text="3) 选择参数").grid(row=row, column=0, sticky="w")
        row += 1
        self.param_form_frame = ttk.Frame(control_panel)
        self.param_form_frame.grid(row=row, column=0, pady=(2, 6), sticky="ew")
        row += 1
        self.lock_params_btn = ttk.Button(
            control_panel, text="锁定参数", command=self.lock_params, state="disabled"
        )
        self.lock_params_btn.grid(row=row, column=0, pady=(0, 10), sticky="ew")

        row += 1
        ttk.Separator(control_panel, orient="horizontal").grid(row=row, column=0, sticky="ew", pady=6)
        row += 1
        self.duel_btn = ttk.Button(
            control_panel,
            text="开始判定",
            command=self.start_or_rematch,
            state="disabled",
        )
        self.duel_btn.grid(row=row, column=0, pady=(2, 6), sticky="ew")
        row += 1
        self.back_param_btn = ttk.Button(
            control_panel, text="返回参数选择", command=self.back_to_params, state="disabled"
        )
        self.back_param_btn.grid(row=row, column=0, pady=2, sticky="ew")
        row += 1
        self.back_model_btn = ttk.Button(
            control_panel, text="返回模型选择", command=self.back_to_model, state="disabled"
        )
        self.back_model_btn.grid(row=row, column=0, pady=2, sticky="ew")
        row += 1
        self.back_enemy_btn = ttk.Button(
            control_panel, text="返回敌人选择", command=self.back_to_opponent, state="disabled"
        )
        self.back_enemy_btn.grid(row=row, column=0, pady=2, sticky="ew")

        info_panel = ttk.Frame(main)
        info_panel.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(12, 0))
        info_panel.columnconfigure(0, weight=1)
        info_panel.rowconfigure(1, weight=1)

        dist_frame = ttk.LabelFrame(info_panel, text="分布说明", padding=8)
        dist_frame.grid(row=0, column=0, sticky="ew")
        dist_text = tk.Text(dist_frame, height=9, wrap="word")
        dist_text.grid(row=0, column=0, sticky="ew")
        dist_text.insert("1.0", self._distribution_help_text())
        dist_text.configure(state="disabled")

        log_frame = ttk.LabelFrame(info_panel, text="对战记录", padding=8)
        log_frame.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, wrap="word")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set, state="disabled")

    def _distribution_help_text(self) -> str:
        lines: list[str] = []
        for i, dist in enumerate(self.distributions, start=1):
            lines.append(f"{i}. {dist.name}")
            lines.append(f"   {dist.description}")
            lines.append(f"   {dist.centering_note}")
        return "\n".join(lines)

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_enemy_status(self) -> None:
        if self.enemy_selection is None:
            self.enemy_status_var.set("未锁定")
            return
        self.enemy_status_var.set(
            f"{self.enemy_selection.model.name} ({format_selection_params(self.enemy_selection)})"
        )

    def _set_player_status(self) -> None:
        if self.player_selection is not None:
            self.player_status_var.set(
                f"{self.player_selection.model.name} ({format_selection_params(self.player_selection)})"
            )
        elif self.current_model is not None:
            self.player_status_var.set(f"模型已锁定: {self.current_model.name}，参数未锁定")
        else:
            self.player_status_var.set("未锁定")

    def _update_duel_button_text(self) -> None:
        if self.fixed_total > 0:
            self.duel_btn.configure(text="立刻再次判定（保持敌人、模型、参数不变）")
        else:
            self.duel_btn.configure(text="开始判定")

    def _set_fixed_stats_status(self) -> None:
        if self.fixed_total == 0:
            self.fixed_stats_var.set("未开始判定")
            return

        win_rate = 100.0 * self.fixed_player_wins / self.fixed_total
        self.fixed_stats_var.set(
            f"{self.fixed_player_wins}胜 {self.fixed_enemy_wins}负 {self.fixed_ties}平，"
            f"胜率={win_rate:.2f}%（{self.fixed_player_wins}/{self.fixed_total}）"
        )

    def _reset_fixed_stats(self) -> None:
        self.battle_round = 0
        self.fixed_total = 0
        self.fixed_player_wins = 0
        self.fixed_enemy_wins = 0
        self.fixed_ties = 0
        self._set_fixed_stats_status()
        self._update_duel_button_text()

    def _reset_to_opponent_level(self) -> None:
        self.current_opponent = None
        self.enemy_selection = None
        self.current_model = None
        self.player_selection = None
        self.param_vars = {}
        self._reset_fixed_stats()

        self.opponent_combo.configure(state="readonly")
        if not self.opponent_combo.get() and self.opponents:
            self.opponent_combo.current(0)

        self.model_combo.set("")
        self.model_combo.configure(state="disabled")
        self.lock_model_btn.configure(state="disabled")
        self.lock_params_btn.configure(state="disabled")
        self.duel_btn.configure(state="disabled")
        self.back_param_btn.configure(state="disabled")
        self.back_model_btn.configure(state="disabled")
        self.back_enemy_btn.configure(state="disabled")
        self._clear_param_form()
        self._set_enemy_status()
        self._set_player_status()

    def _clear_param_form(self) -> None:
        for child in self.param_form_frame.winfo_children():
            child.destroy()
        self.param_vars = {}

    def _normalize_param_var_realtime(self, var: tk.StringVar) -> None:
        value = var.get()
        normalized = normalize_numeric_input(value)
        if normalized != value:
            var.set(normalized)

    def _is_period_key_event(self, event: tk.Event) -> bool:
        if event.keysym in PERIOD_KEYSYMS:
            return True
        keycode = getattr(event, "keycode", None)
        if isinstance(keycode, int) and keycode in PERIOD_KEYCODES:
            return True
        return False

    def _on_param_keypress(self, event: tk.Event) -> str | None:
        # Ignore shortcuts such as Ctrl+.
        if event.state & 0x000C:
            return None

        if event.char == "." or event.char in DOT_ALIASES or self._is_period_key_event(event):
            event.widget.insert(tk.INSERT, ".")
            return "break"
        if event.char == "-" or event.char in MINUS_ALIASES:
            event.widget.insert(tk.INSERT, "-")
            return "break"
        return None

    def _build_param_form(self, model: DistributionSpec, preset: dict[str, Any] | None = None) -> None:
        self._clear_param_form()
        for i, spec in enumerate(model.params):
            ttk.Label(self.param_form_frame, text=spec.prompt).grid(row=i, column=0, sticky="w", pady=1)
            value = ""
            if preset and spec.name in preset:
                value = str(preset[spec.name])
            var = tk.StringVar(value=value)
            self.param_vars[spec.name] = var
            var.trace_add("write", lambda *_args, v=var: self._normalize_param_var_realtime(v))
            entry = ttk.Entry(self.param_form_frame, textvariable=var, width=20)
            entry.bind("<KeyPress>", self._on_param_keypress)
            entry.grid(
                row=i, column=1, sticky="ew", padx=(8, 0), pady=1
            )

    def _read_params(self, model: DistributionSpec) -> dict[str, Any] | None:
        values: dict[str, Any] = {}
        for spec in model.params:
            raw = self.param_vars.get(spec.name, tk.StringVar(value="")).get().strip()
            try:
                value = spec.caster(raw)
            except ValueError:
                messagebox.showerror("参数错误", f"{spec.name} 输入格式错误，请重新输入。")
                return None
            error = spec.validator(value)
            if error:
                messagebox.showerror("参数错误", f"{spec.name} 无效: {error}")
                return None
            values[spec.name] = value

        joint_error = model.joint_validator(values)
        if joint_error:
            messagebox.showerror("参数错误", f"参数组合无效: {joint_error}")
            return None
        return values

    def lock_opponent(self) -> None:
        idx = self.opponent_combo.current()
        if idx < 0:
            messagebox.showwarning("提示", "请先选择敌人。")
            return

        self.current_opponent = self.opponents[idx]
        self.enemy_selection = self.current_opponent.strategy(self.rng)
        self.current_model = None
        self.player_selection = None
        self.param_vars = {}
        self._reset_fixed_stats()

        self.model_combo.configure(state="readonly")
        if self.distributions:
            self.model_combo.current(0)
        self.lock_model_btn.configure(state="normal")
        self.lock_params_btn.configure(state="disabled")
        self.duel_btn.configure(state="disabled")
        self.back_param_btn.configure(state="disabled")
        self.back_model_btn.configure(state="disabled")
        self.back_enemy_btn.configure(state="disabled")
        self._clear_param_form()
        self._set_enemy_status()
        self._set_player_status()
        self._append_log(f"[系统] 敌人已锁定：{self.current_opponent.name}")

    def lock_model(self) -> None:
        if self.enemy_selection is None:
            messagebox.showwarning("提示", "请先锁定敌人。")
            return

        idx = self.model_combo.current()
        if idx < 0:
            messagebox.showwarning("提示", "请先选择模型。")
            return

        self.current_model = self.distributions[idx]
        self.player_selection = None
        self._reset_fixed_stats()
        self._build_param_form(self.current_model)
        self.lock_params_btn.configure(state="normal")
        self.duel_btn.configure(state="disabled")
        self.back_model_btn.configure(state="normal")
        self.back_enemy_btn.configure(state="normal")
        self._set_player_status()
        self._append_log(f"[系统] 模型已锁定：{self.current_model.name}")

    def lock_params(self) -> None:
        if self.current_model is None:
            messagebox.showwarning("提示", "请先锁定模型。")
            return

        parsed = self._read_params(self.current_model)
        if parsed is None:
            return

        self._reset_fixed_stats()
        self.player_selection = Selection(self.current_model, parsed)
        self.duel_btn.configure(state="normal")
        self.back_param_btn.configure(state="normal")
        self.back_model_btn.configure(state="normal")
        self.back_enemy_btn.configure(state="normal")
        self._set_player_status()
        self._append_log("[系统] 参数已锁定，可开始判定。")

    def start_or_rematch(self) -> None:
        if self.player_selection is None or self.enemy_selection is None:
            messagebox.showwarning("提示", "请先锁定敌人、模型和参数。")
            return

        is_instant_rematch = self.fixed_total > 0
        self.battle_round += 1
        result = run_duel(self.rng, self.player_selection, self.enemy_selection)
        self._append_log(f"\n=== 第 {self.battle_round} 局 ===")
        self._append_log(
            f"你：{self.player_selection.model.name} ({format_selection_params(self.player_selection)})"
        )
        self._append_log(
            f"电脑：{self.enemy_selection.model.name} ({format_selection_params(self.enemy_selection)})"
        )
        self._append_log(
            "你："
            f"原始采样={format_number(result.player_raw)}, "
            f"理论期望={format_number(result.player_mean)}, "
            f"判定值={format_number(result.player_score)}"
        )
        self._append_log(
            "电脑："
            f"原始采样={format_number(result.enemy_raw)}, "
            f"理论期望={format_number(result.enemy_mean)}, "
            f"判定值={format_number(result.enemy_score)}"
        )
        self._append_log(f"结果：{result.winner_text}")

        self.fixed_total += 1
        if result.winner_text == "你获胜":
            self.fixed_player_wins += 1
        elif result.winner_text == "电脑获胜":
            self.fixed_enemy_wins += 1
        else:
            self.fixed_ties += 1
        self._set_fixed_stats_status()
        self._update_duel_button_text()

        if is_instant_rematch:
            win_rate = 100.0 * self.fixed_player_wins / self.fixed_total
            self._append_log(
                "[固定配置统计] "
                f"{self.fixed_player_wins}胜 {self.fixed_enemy_wins}负 {self.fixed_ties}平，"
                f"玩家胜率={win_rate:.2f}%（{self.fixed_player_wins}/{self.fixed_total}）"
            )

    def back_to_params(self) -> None:
        if self.current_model is None:
            return
        preset = self.player_selection.params if self.player_selection else None
        self.player_selection = None
        self._reset_fixed_stats()
        self._build_param_form(self.current_model, preset)
        self.duel_btn.configure(state="disabled")
        self._set_player_status()
        self._append_log("[系统] 已返回参数选择。敌人和模型保持不变。")

    def back_to_model(self) -> None:
        if self.enemy_selection is None:
            return
        self.current_model = None
        self.player_selection = None
        self._reset_fixed_stats()
        self.model_combo.configure(state="readonly")
        self.lock_model_btn.configure(state="normal")
        self.lock_params_btn.configure(state="disabled")
        self.duel_btn.configure(state="disabled")
        self.back_param_btn.configure(state="disabled")
        self.back_model_btn.configure(state="disabled")
        self.back_enemy_btn.configure(state="normal")
        self._clear_param_form()
        self._set_player_status()
        self._append_log("[系统] 已返回模型选择。敌人保持不变。")

    def back_to_opponent(self) -> None:
        self._reset_to_opponent_level()
        self._append_log("[系统] 已返回敌人选择。模型和参数已重置。")

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = ProbabilityBattleApp()
    app.run()


if __name__ == "__main__":
    main()
