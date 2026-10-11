"""Честная проверка на собственных записях: обучение без одного дня, проверка на нём.

Своих записей мало, и все они сняты одним человеком на одном телефоне,
поэтому случайное деление окон ничего не говорит: соседние окна одной записи
почти одинаковы. Делить нужно по дням — запись нового дня ближе всего к тому,
что модель увидит в приложении завтра.

Для каждого дня D: модели обучаются на публичных датасетах и своих записях
всех дней, кроме D, экспортируются в .tflite и прогоняются по записям D —
ровно тем файлом, что поедет в APK. Итог — полнота по каждому сочетанию
«активность × положение» и средняя полнота по классам.

С --compare-assets те же окна прогоняются через другую модель (например,
текущую из приложения), чтобы сравнение шло на одинаковых данных.

Запуск:
    python ml/crossval.py --source all
    python ml/crossval.py --days 20261007,20261009 --compare-assets путь/к/старым/assets
Остальные параметры — как у train.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import datasets
import evaluate
import train


def _recalls(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, tuple[int, int]]:
    """Класс → (верно, всего)."""
    return {c: (int(((y_true == c) & (y_pred == c)).sum()), int((y_true == c).sum()))
            for c in np.unique(y_true)}


def _predict(assets: Path, own: datasets.Dataset) -> tuple[np.ndarray | None, np.ndarray | None]:
    meta = json.loads((assets / "model_meta.json").read_text(encoding="utf-8"))
    act = evaluate.predict_activity(meta, own, assets)
    pl = evaluate.predict_placement(meta, own, assets)
    a = None if act is None else np.array(meta["activity_labels"])[act.argmax(axis=1)]
    p = None if pl is None else np.array(meta["placement_labels"])[pl.argmax(axis=1)]
    return a, p


def _table(title: str, rows: dict[str, dict[str, tuple[int, int]]], models: list[str]) -> None:
    print(f"\n{title}")
    print(f"  {'':<28}" + "".join(f"{m:>14}" for m in models))
    keys = sorted({k for r in rows.values() for k in r})
    macro = {m: [] for m in models}
    for k in keys:
        cells = []
        for m in models:
            ok, n = rows[m].get(k, (0, 0))
            cells.append(f"{ok / n:>8.0%} ({n:>4})" if n else f"{'—':>14}")
            if n:
                macro[m].append(ok / n)
        print(f"  {k:<28}" + "".join(cells))
    print(f"  {'среднее по строкам':<28}" + "".join(
        f"{np.mean(macro[m]):>14.1%}" if macro[m] else f"{'—':>14}" for m in models))


def main() -> None:
    parser = train.build_parser()
    parser.description = __doc__
    parser.add_argument("--days", default="", help="какие дни проверять; по умолчанию все")
    parser.add_argument("--compare-assets", type=Path, default=None,
                        help="каталог с другой моделью для сравнения на тех же окнах")
    parser.add_argument("--out-dir", type=Path, default=train.REPO_ROOT / "ml" / "runs" / "crossval",
                        help="куда класть модели фолдов и итоговый JSON")
    args = parser.parse_args()
    train.setup(args)

    wanted = train.parse_sources(args.source)
    if "own" not in wanted:
        raise SystemExit("Для проверки по дням нужны собственные записи (источник own)")
    public = train.load_public(wanted, args)

    all_days = sorted({datasets.own_day(p.name) for p in args.own_dir.glob("*.csv")})
    days = [d for d in args.days.split(",") if d] or all_days
    print(f"\nДни собственных записей: {', '.join(all_days)}; проверяем: {', '.join(days)}")

    models = ["новая"] + (["сравнение"] if args.compare_assets else [])
    act_rows = {m: {} for m in models}
    pl_rows = {m: {} for m in models}
    combo_rows = {m: {} for m in models}
    per_day = {}

    def add(store: dict, key: str, ok: int, n: int) -> None:
        a, b = store.get(key, (0, 0))
        store[key] = (a + ok, b + n)

    for day in days:
        print(f"\n{'=' * 70}\nДень {day}: обучение без него\n{'=' * 70}")
        own_train = train.load_own_filtered(args, exclude_days={day})
        own_test = datasets.load_own(args.own_dir, only_days={day})
        if own_test is None:
            continue
        sources = public + ([own_train] if own_train is not None else [])
        fold_dir = args.out_dir / day
        activity, placement, meta = train.train_models(sources, args)
        train.write_assets(fold_dir, activity, placement, meta)

        preds = {"новая": _predict(fold_dir, own_test)}
        if args.compare_assets:
            preds["сравнение"] = _predict(args.compare_assets, own_test)

        combo = np.char.add(np.char.add(own_test.y_activity.astype(str), " / "),
                            own_test.y_placement.astype(str))
        per_day[day] = {}
        for m, (a, p) in preds.items():
            day_rows = {}
            if a is not None:
                for c, (ok, n) in _recalls(own_test.y_activity, a).items():
                    add(act_rows[m], c, ok, n)
                for c in np.unique(combo):
                    mask = combo == c
                    ok, n = int((a[mask] == own_test.y_activity[mask]).sum()), int(mask.sum())
                    add(combo_rows[m], c, ok, n)
                    day_rows[c] = (ok, n)
            if p is not None:
                for c, (ok, n) in _recalls(own_test.y_placement, p).items():
                    add(pl_rows[m], c, ok, n)
            per_day[day][m] = {
                "activity_combo": day_rows,
                "placement": _recalls(own_test.y_placement, p) if p is not None else {},
                "activity_confusion": {
                    c: dict(zip(*np.unique(a[combo == c], return_counts=True)))
                    for c in np.unique(combo)
                } if a is not None else {},
            }
        print(f"\nДень {day}:")
        _table("Активность по записям (полнота):",
               {m: per_day[day][m]["activity_combo"] for m in models}, models)

    print(f"\n{'=' * 70}\nИТОГ ПО ВСЕМ ДНЯМ (каждое окно проверено моделью, не видевшей его день)\n{'=' * 70}")
    _table("Активность × положение (полнота активности):", combo_rows, models)
    _table("Активность (полнота):", act_rows, models)
    _table("Положение телефона (полнота):", pl_rows, models)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "summary.json"

    def plain(o):
        if isinstance(o, dict):
            return {str(k): plain(v) for k, v in o.items()}
        if isinstance(o, (tuple, list)):
            return [plain(v) for v in o]
        return o.item() if hasattr(o, "item") else o

    out.write_text(json.dumps(plain({"activity_combo": combo_rows, "activity": act_rows,
                                     "placement": pl_rows, "per_day": per_day}),
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nПодробности: {out}")


if __name__ == "__main__":
    main()
