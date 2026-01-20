from __future__ import annotations

import csv
import io
import itertools
import os
import random
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- Загрузка данных из sqlite ----------

def load_ballots_sqlite(db_path: str = "bot.sqlite3") -> List[List[int]]:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    query = """
            SELECT ranking_json
            FROM ballots b1
            WHERE id = (
                SELECT MAX(id)
                FROM ballots b2
                WHERE b2.user_id = b1.user_id
            )
            ORDER BY id ASC
        """
    cur.execute(query)
    rows = cur.fetchall()
    con.close()

    import json
    ballots = [json.loads(r[0]) for r in rows]
    # нормализуем в list[int]
    ballots = [list(map(int, b)) for b in ballots if b]
    return ballots


def load_options_sqlite(db_path: str = "bot.sqlite3") -> Tuple[List[int], Dict[int, str]]:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("SELECT id, title FROM options ORDER BY id ASC")
    rows = cur.fetchall()
    con.close()
    candidate_ids = [int(r[0]) for r in rows]
    labels = {int(r[0]): str(r[1]) for r in rows}
    return candidate_ids, labels


# ---------- IRV: порядок выбывания + раунды ----------

def irv_rounds(ballots: Sequence[Sequence[int]], candidate_ids: Sequence[int]) -> List[dict]:
    active = set(candidate_ids)
    rounds: List[dict] = []
    rnd = 1

    while True:
        counts = {cid: 0 for cid in active}
        exhausted = 0

        for ballot in ballots:
            choice = None
            for pref in ballot:
                if pref in active:
                    choice = pref
                    break
            if choice is None:
                exhausted += 1
            else:
                counts[choice] += 1

        total_active_votes = sum(counts.values())
        info = {
            "round": rnd,
            "active": sorted(active),
            "counts": dict(sorted(counts.items(), key=lambda x: (-x[1], x[0]))),
            "exhausted": exhausted,
            "total_active_votes": total_active_votes,
        }

        if len(active) == 1:
            winner = next(iter(active))
            info["winner"] = winner
            rounds.append(info)
            break

        # победа по большинству среди активных
        if total_active_votes > 0:
            for cid, v in counts.items():
                if v > total_active_votes / 2:
                    info["winner"] = cid
                    rounds.append(info)
                    active = {cid}
                    break
            if len(active) == 1:
                break

        # выбывает минимум (tie -> min id)
        min_votes = min(counts.values()) if counts else 0
        losers = [cid for cid, v in counts.items() if v == min_votes]
        eliminated = min(losers)

        info["eliminated"] = eliminated
        rounds.append(info)
        active.remove(eliminated)
        rnd += 1

    return rounds


def irv_order_from_rounds(rounds: List[dict]) -> List[int]:
    # rounds содержат eliminated по шагам, winner в последнем
    eliminated = [r["eliminated"] for r in rounds if "eliminated" in r]
    winner = rounds[-1].get("winner")
    if winner is None:
        # если winner не проставился (не должно), берём последнего активного
        winner = rounds[-1]["active"][0]
    # Хотим: winner ... почти победитель ... самый слабый
    return [winner] + eliminated[::-1]


# ---------- Парные сравнения: wins / non_ties / share ----------

def pairwise_wins_non_ties(
    ballots: Sequence[Sequence[int]],
    ordered_ids: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Возвращает wins[r,c], non_ties[r,c], share[r,c] (по ordered_ids индексам).
    Вариант, которого нет в бюллетене, проигрывает любому, который есть.
    Если оба отсутствуют — ничья (не считаем в non_ties).
    """
    n = len(ordered_ids)
    wins = np.zeros((n, n), dtype=int)
    non_ties = np.zeros((n, n), dtype=int)

    pos_maps: List[Dict[int, int]] = [{cid: i for i, cid in enumerate(b)} for b in ballots]

    for r, a in enumerate(ordered_ids):
        for c, b in enumerate(ordered_ids):
            if r == c:
                continue
            w = 0
            nt = 0
            for pm in pos_maps:
                in_a = a in pm
                in_b = b in pm
                if not in_a and not in_b:
                    continue
                nt += 1
                if in_a and in_b:
                    if pm[a] < pm[b]:
                        w += 1
                else:
                    if in_a and not in_b:
                        w += 1
            wins[r, c] = w
            non_ties[r, c] = nt

    share = np.full((n, n), 0.5, dtype=float)
    for r in range(n):
        for c in range(n):
            if r == c:
                continue
            nt = non_ties[r, c]
            share[r, c] = wins[r, c] / nt if nt > 0 else 0.5

    return wins, non_ties, share


# ---------- Метрики кандидатов ----------

def first_choice_counts(ballots: Sequence[Sequence[int]], candidate_ids: Sequence[int]) -> Dict[int, int]:
    out = {cid: 0 for cid in candidate_ids}
    for b in ballots:
        if b:
            out[b[0]] += 1
    return out


def presence_counts(ballots: Sequence[Sequence[int]], candidate_ids: Sequence[int]) -> Dict[int, int]:
    out = {cid: 0 for cid in candidate_ids}
    for b in ballots:
        s = set(b)
        for cid in candidate_ids:
            if cid in s:
                out[cid] += 1
    return out


def topk_counts(ballots: Sequence[Sequence[int]], candidate_ids: Sequence[int], k: int) -> Dict[int, int]:
    out = {cid: 0 for cid in candidate_ids}
    for b in ballots:
        top = set(b[:k])
        for cid in top:
            if cid in out:
                out[cid] += 1
    return out


def rank_stats_present(ballots: Sequence[Sequence[int]], candidate_ids: Sequence[int]) -> Dict[int, dict]:
    """
    Статистика ранга среди тех бюллетеней, где кандидат присутствует.
    Ранги считаем 1..len(ballot).
    """
    ranks = {cid: [] for cid in candidate_ids}
    for b in ballots:
        pm = {cid: i for i, cid in enumerate(b)}
        for cid in candidate_ids:
            if cid in pm:
                ranks[cid].append(pm[cid] + 1)

    out: Dict[int, dict] = {}
    for cid, arr in ranks.items():
        if not arr:
            out[cid] = {"mean": None, "median": None, "n": 0}
        else:
            a = np.array(arr)
            out[cid] = {"mean": float(a.mean()), "median": float(np.median(a)), "n": int(len(a)), "all": arr}
    return out


def borda_scores(ballots: Sequence[Sequence[int]], candidate_ids: Sequence[int], m: int) -> Dict[int, int]:
    """
    Borda: в каждом бюллетене кандидат на позиции p (0..L-1) получает (m-1-p) очков.
    Отсутствующие получают 0.
    """
    score = {cid: 0 for cid in candidate_ids}
    for b in ballots:
        for p, cid in enumerate(b):
            if cid in score:
                score[cid] += (m - 1 - p)
    return score


def copeland_scores_from_share(share: np.ndarray, non_ties: np.ndarray) -> np.ndarray:
    """
    Copeland = число парных побед - число парных поражений.
    Ничьи (share==0.5) считаем 0.
    """
    n = share.shape[0]
    s = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(n):
            if i == j or non_ties[i, j] == 0:
                continue
            if share[i, j] > 0.5:
                s[i] += 1
            elif share[i, j] < 0.5:
                s[i] -= 1
    return s


# ---------- Condorcet / Smith set (брютфорс для n=15) ----------

def condorcet_winner(share: np.ndarray, non_ties: np.ndarray, ids: Sequence[int]) -> Optional[int]:
    n = len(ids)
    for i in range(n):
        ok = True
        for j in range(n):
            if i == j:
                continue
            if non_ties[i, j] == 0:
                ok = False
                break
            if share[i, j] <= 0.5:
                ok = False
                break
        if ok:
            return ids[i]
    return None


def smith_set(share: np.ndarray, non_ties: np.ndarray, ids: Sequence[int]) -> List[int]:
    """
    Ищем минимальный набор S такой, что каждый i в S побеждает каждого j вне S (majority).
    Для n=15 можно спокойно перебрать подмножества по возрастанию размера.
    """
    n = len(ids)
    beats = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            beats[i, j] = (non_ties[i, j] > 0 and share[i, j] > 0.5)

    idx = list(range(n))
    for size in range(1, n + 1):
        for comb in itertools.combinations(idx, size):
            S = set(comb)
            good = True
            for i in S:
                for j in idx:
                    if j in S:
                        continue
                    if not beats[i, j]:
                        good = False
                        break
                if not good:
                    break
            if good:
                return [ids[i] for i in sorted(S)]
    return list(ids)


# ---------- Визуализации ----------

def _label(cid: int, labels: Dict[int, str]) -> str:
    return labels.get(cid, str(cid))


def plot_bar(order: List[int], values: Dict[int, float], labels: Dict[int, str], title: str, path: str):
    x = np.arange(len(order))
    y = [values.get(cid, 0) for cid in order]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x, y)
    ax.set_xticks(x)
    ax.set_xticklabels([_label(cid, labels) for cid in order], rotation=90)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_topk_heatmap(order: List[int], topk: Dict[int, List[int]], labels: Dict[int, str], ks: List[int], path: str):
    # матрица: кандидаты × ks, значения = доля (0..1)
    mat = np.zeros((len(order), len(ks)), dtype=float)
    n_ballots = None
    for r, cid in enumerate(order):
        for c, k in enumerate(ks):
            cnt = topk[k][cid]
            if n_ballots is None:
                # n_ballots можно восстановить по максимуму присутствия? лучше передать, но не критично
                pass
            mat[r, c] = cnt

    # нормируем по max (чтобы было похоже на доли — но лучше отдать доли сразу)
    # здесь ожидаем, что topk[k][cid] — это count; превращаем в долю по сумме бюллетеней:
    # (передадим n_ballots извне в generate_report, там пересчитаем на долю)

    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(mat, aspect="auto")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([_label(cid, labels) for cid in order])
    ax.set_xticks(np.arange(len(ks)))
    ax.set_xticklabels([f"Top-{k}" for k in ks])
    ax.set_title("Top-k counts (raw)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_topk_heatmap_shares(order: List[int], topk_counts_by_k: Dict[int, Dict[int, int]], labels: Dict[int, str],
                            ks: List[int], n_ballots: int, path: str):
    mat = np.zeros((len(order), len(ks)), dtype=float)
    for r, cid in enumerate(order):
        for c, k in enumerate(ks):
            mat[r, c] = topk_counts_by_k[k][cid] / n_ballots if n_ballots else 0.0

    fig, ax = plt.subplots(figsize=(7, 10))
    im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([_label(cid, labels) for cid in order])
    ax.set_xticks(np.arange(len(ks)))
    ax.set_xticklabels([f"Top-{k}" for k in ks])
    ax.set_title("Top-k доли")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_rank_boxplot(order: List[int], rank_stats: Dict[int, dict], labels: Dict[int, str], path: str):
    data = []
    tick = []
    for cid in order:
        arr = rank_stats[cid].get("all", [])
        if arr:
            data.append(arr)
        else:
            data.append([np.nan])
        tick.append(_label(cid, labels))

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.boxplot(data, showfliers=False)
    ax.set_xticklabels(tick, rotation=90)
    ax.set_title("Распределение рангов (среди бюллетеней, где кандидат присутствует)")
    ax.set_ylabel("Ранг (1 = лучший)")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ballot_length_hist(ballots: Sequence[Sequence[int]], path: str):
    lens = [len(b) for b in ballots]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lens, bins=range(0, max(lens) + 2))
    ax.set_title("Длины рейтингов")
    ax.set_xlabel("Сколько вариантов ранжировано")
    ax.set_ylabel("Количество бюллетеней")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_irv_active_exhausted(rounds: List[dict], path: str):
    rs = [r["round"] for r in rounds]
    active_votes = [r["total_active_votes"] for r in rounds]
    exhausted = [r["exhausted"] for r in rounds]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rs, active_votes, marker="o", label="active votes")
    ax.plot(rs, exhausted, marker="o", label="exhausted")
    ax.set_title("IRV по раундам: active vs exhausted")
    ax.set_xlabel("Раунд")
    ax.set_ylabel("Количество бюллетеней")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bootstrap_winners(ballots: List[List[int]], candidate_ids: List[int], labels: Dict[int, str], B: int, path: str):
    # bootstrap: семплируем бюллетени с возвращением, считаем IRV winner
    n = len(ballots)
    rng = random.Random(0)
    win_counts = {cid: 0 for cid in candidate_ids}

    for _ in range(B):
        sample = [ballots[rng.randrange(n)] for _ in range(n)]
        r = irv_rounds(sample, candidate_ids)
        winner = r[-1].get("winner", r[-1]["active"][0])
        win_counts[winner] += 1

    # рисуем
    order = sorted(candidate_ids, key=lambda cid: (-win_counts[cid], cid))
    x = np.arange(len(order))
    y = [win_counts[cid] / B for cid in order]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x, y)
    ax.set_xticks(x)
    ax.set_xticklabels([_label(cid, labels) for cid in order], rotation=90)
    ax.set_ylim(0, 1)
    ax.set_title(f"Bootstrap IRV winners (B={B}) — доля побед")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------- CSV helpers ----------

def write_csv(path: str, header: List[str], rows: List[List[object]]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ---------- ТВОЯ 15×15 матрица (если хочешь включить) ----------
# Вставь сюда функцию plot_irv_pairwise_matrix(ballots, candidate_ids, labels, ...)
# и в generate_report включи вызов.


def generate_report(db_path: str = "bot.sqlite3", out_dir: str = "report"):
    os.makedirs(out_dir, exist_ok=True)

    ballots = load_ballots_sqlite(db_path)
    candidate_ids, labels = load_options_sqlite(db_path)

    if not ballots:
        raise RuntimeError("Нет бюллетеней в ballots.")

    n_ballots = len(ballots)
    m = len(candidate_ids)

    # IRV rounds + order
    rounds = irv_rounds(ballots, candidate_ids)
    order = irv_order_from_rounds(rounds)

    # pairwise matrices in IRV order
    wins, non_ties, share = pairwise_wins_non_ties(ballots, order)

    # basic counts
    fc = first_choice_counts(ballots, candidate_ids)
    pres = presence_counts(ballots, candidate_ids)
    top3 = topk_counts(ballots, candidate_ids, 3)
    top5 = topk_counts(ballots, candidate_ids, 5)
    rstats = rank_stats_present(ballots, candidate_ids)
    borda = borda_scores(ballots, candidate_ids, m)

    # Copeland по IRV-порядку
    cop = copeland_scores_from_share(share, non_ties)
    cop_by_id = {order[i]: int(cop[i]) for i in range(len(order))}

    # Condorcet / Smith (в IRV order индексах)
    cw = condorcet_winner(share, non_ties, order)
    ss = smith_set(share, non_ties, order)

    print(f"Всего бюллетеней: {n_ballots}")
    if cw is not None:
        print(f"Condorcet winner: {cw} — {_label(cw, labels)}")
    else:
        print("Condorcet winner: нет")
    print("Smith set:", [f"{cid}:{_label(cid, labels)}" for cid in ss])

    # ----- CSV: IRV rounds -----
    irv_rows = []
    for r in rounds:
        eliminated = r.get("eliminated", "")
        winner = r.get("winner", "")
        # counts в одну строку "id:count; ..."
        counts_str = "; ".join([f"{cid}:{v}" for cid, v in r["counts"].items()])
        irv_rows.append([r["round"], r["total_active_votes"], r["exhausted"], eliminated, winner, counts_str])

    write_csv(
        os.path.join(out_dir, "irv_rounds.csv"),
        ["round", "active_votes", "exhausted", "eliminated", "winner", "counts"],
        irv_rows,
    )

    # ----- CSV: pairwise share -----
    pw_rows = []
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            if i == j:
                continue
            pw_rows.append([
                a, _label(a, labels),
                b, _label(b, labels),
                int(wins[i, j]),
                int(non_ties[i, j]),
                float(share[i, j]),
            ])

    write_csv(
        os.path.join(out_dir, "pairwise_share.csv"),
        ["row_id", "row_label", "col_id", "col_label", "wins", "non_ties", "share"],
        pw_rows,
    )

    # ----- CSV: co-occurrence -----
    # доля бюллетеней, где оба присутствуют
    co = np.zeros((m, m), dtype=float)
    idx = {cid: k for k, cid in enumerate(candidate_ids)}
    for b in ballots:
        s = set(b)
        for a in s:
            for c in s:
                if a in idx and c in idx:
                    co[idx[a], idx[c]] += 1
    co /= n_ballots

    co_rows = []
    for a in candidate_ids:
        for b in candidate_ids:
            co_rows.append([a, _label(a, labels), b, _label(b, labels), float(co[idx[a], idx[b]])])

    write_csv(
        os.path.join(out_dir, "cooccurrence.csv"),
        ["a_id", "a_label", "b_id", "b_label", "share_both_present"],
        co_rows,
    )

    # ----- CSV: summary metrics -----
    # exit round: когда выбыл; победитель = None (или последняя стадия)
    exit_round = {cid: None for cid in candidate_ids}
    for r in rounds:
        if "eliminated" in r:
            exit_round[r["eliminated"]] = r["round"]

    winner = rounds[-1].get("winner", rounds[-1]["active"][0])
    exit_round[winner] = rounds[-1]["round"]

    summary_rows = []
    for cid in order:
        mean_rank = rstats[cid]["mean"]
        med_rank = rstats[cid]["median"]
        summary_rows.append([
            cid,
            _label(cid, labels),
            fc.get(cid, 0),
            pres.get(cid, 0),
            top3.get(cid, 0),
            top5.get(cid, 0),
            mean_rank if mean_rank is not None else "",
            med_rank if med_rank is not None else "",
            borda.get(cid, 0),
            cop_by_id.get(cid, 0),
            exit_round.get(cid, ""),
        ])

    write_csv(
        os.path.join(out_dir, "summary_metrics.csv"),
        ["id", "label", "top1", "present", "top3", "top5", "mean_rank_present", "median_rank_present",
         "borda", "copeland", "irv_round"],
        summary_rows,
    )

    # ----- Plots -----
    plot_bar(order, fc, labels, f"Top-1 (plurality) — N={n_ballots}", os.path.join(out_dir, "01_plurality_first_choices.png"))
    plot_bar(order, pres, labels, f"Присутствует в бюллетенях — N={n_ballots}", os.path.join(out_dir, "02_presence.png"))
    plot_bar(order, borda, labels, f"Borda score — N={n_ballots}", os.path.join(out_dir, "03_borda.png"))
    plot_bar(order, cop_by_id, labels, f"Copeland score (pairwise strength) — N={n_ballots}", os.path.join(out_dir, "04_copeland.png"))

    topk_counts_by_k = {
        1: fc,
        2: topk_counts(ballots, candidate_ids, 2),
        3: top3,
        4: topk_counts(ballots, candidate_ids, 4),
        5: top5,
    }
    plot_topk_heatmap_shares(order, topk_counts_by_k, labels, ks=[1,2,3,4,5], n_ballots=n_ballots,
                            path=os.path.join(out_dir, "05_topk_heatmap_1_5.png"))

    plot_rank_boxplot(order, rstats, labels, os.path.join(out_dir, "06_rank_boxplot_present.png"))
    plot_ballot_length_hist(ballots, os.path.join(out_dir, "07_ballot_length_hist.png"))
    plot_irv_active_exhausted(rounds, os.path.join(out_dir, "08_irv_rounds_active_exhausted.png"))

    plot_bootstrap_winners(ballots, candidate_ids, labels, B=500, path=os.path.join(out_dir, "09_bootstrap_irv_winners.png"))

    # Если вставишь свою plot_irv_pairwise_matrix — раскомментируй:
    # fig, ax = plot_irv_pairwise_matrix(ballots, candidate_ids=candidate_ids, labels=labels, figsize=(13,11))
    # fig.savefig(os.path.join(out_dir, "10_pairwise_matrix.png"), dpi=200, bbox_inches="tight")
    # plt.close(fig)

    print(f"Готово. Смотри папку: {out_dir}/")


if __name__ == "__main__":
    generate_report("bot.sqlite3", "report")
