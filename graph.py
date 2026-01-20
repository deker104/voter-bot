from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt


def _irv_elimination_order(
    ballots: Sequence[Sequence[int]],
    candidate_ids: Sequence[int],
) -> Tuple[int, List[int]]:
    """
    Возвращает (winner_id, eliminated_in_order)
    eliminated_in_order: кандидаты в порядке выбывания (первый = худший, последний = почти победитель).
    Тайбрейк при равенстве: минимальный id (детерминированно).
    """
    active = set(candidate_ids)
    eliminated: List[int] = []

    while len(active) > 1:
        counts = {cid: 0 for cid in active}

        # считаем первые активные предпочтения
        for ballot in ballots:
            chosen = None
            for pref in ballot:
                if pref in active:
                    chosen = pref
                    break
            if chosen is not None:
                counts[chosen] += 1

        # выбывает тот, у кого минимум голосов (tie -> min id)
        min_votes = min(counts.values())
        losers = [cid for cid, v in counts.items() if v == min_votes]
        out = min(losers)

        active.remove(out)
        eliminated.append(out)

    winner = next(iter(active))
    return winner, eliminated


def plot_irv_pairwise_matrix(
    ballots: Sequence[Sequence[int]],
    candidate_ids: Optional[Sequence[int]] = None,
    labels: Optional[Dict[int, str]] = None,
    figsize: Tuple[int, int] = (13, 11),
):
    """
    ballots: список рейтингов, каждый рейтинг — список candidate_id в порядке предпочтения (лучший -> хуже).
             Можно, чтобы некоторые кандидаты отсутствовали в рейтинге.
    candidate_ids: список всех кандидатов (по умолчанию берётся из union всех ballot).
                   Для вашей задачи обычно это 15 ids.
    labels: опционально отображаемые имена вместо id.
    """
    ballots = [list(b) for b in ballots]
    n_ballots = len(ballots)

    if candidate_ids is None:
        all_ids = sorted({cid for b in ballots for cid in b})
        candidate_ids = all_ids

    candidate_ids = list(candidate_ids)
    m = len(candidate_ids)
    if m != 15:
        # не критично, но вы просили 15×15 — пусть будет честное предупреждение
        print(f"⚠️ Внимание: кандидатів {m}, матрица будет {m}×{m}.")

    # 1) сортируем от победителя к проигравшему по порядку выбывания в IRV
    winner, eliminated = _irv_elimination_order(ballots, candidate_ids)
    order = [winner] + eliminated[::-1]  # winner, затем "почти победитель"... затем первый вылетевший (самый слабый)

    # Быстрый доступ: позиция кандидата в бюллетене
    # pos[ballot_index][candidate_id] = rank_index (0..)
    pos_maps: List[Dict[int, int]] = []
    for b in ballots:
        pos_maps.append({cid: i for i, cid in enumerate(b)})

    # 3) диагональ: top1_count / present_count
    top1_count = {cid: 0 for cid in candidate_ids}
    present_count = {cid: 0 for cid in candidate_ids}

    for i, b in enumerate(ballots):
        if b:
            top1_count[b[0]] += 1
        pm = pos_maps[i]
        for cid in candidate_ids:
            if cid in pm:
                present_count[cid] += 1

    # 4) вне диагонали: wins / non_ties + цвет по доле wins/non_ties
    # tie = оба отсутствуют в рейтинге
    wins = np.zeros((m, m), dtype=int)
    non_ties = np.zeros((m, m), dtype=int)

    # предвычислим индексы кандидатов в "order"
    for r, cid_r in enumerate(order):
        for c, cid_c in enumerate(order):
            if r == c:
                continue
            w = 0
            nt = 0
            for bi in range(n_ballots):
                pm = pos_maps[bi]
                in_r = cid_r in pm
                in_c = cid_c in pm
                if not in_r and not in_c:
                    # ничья (оба отсутствуют) -> не считаем
                    continue
                nt += 1
                if in_r and in_c:
                    # кто выше (меньше индекс) — тот победил
                    if pm[cid_r] < pm[cid_c]:
                        w += 1
                else:
                    # вариант, которого нет, проигрывает любому, который есть
                    if in_r and not in_c:
                        w += 1
            wins[r, c] = w
            non_ties[r, c] = nt

    # доли для цветов (где nt=0, ставим 0.5, но это будет редкость)
    share = np.full((m, m), 0.5, dtype=float)
    mask_diag = np.eye(m, dtype=bool)
    for r in range(m):
        for c in range(m):
            if r == c:
                continue
            nt = non_ties[r, c]
            share[r, c] = (wins[r, c] / nt) if nt > 0 else 0.5

    # 2) таблица 15×15, строки/столбцы по order
    # аннотации
    ann = np.empty((m, m), dtype=object)
    for r, cid_r in enumerate(order):
        for c, cid_c in enumerate(order):
            if r == c:
                ann[r, c] = f"{top1_count[cid_r]}/{present_count[cid_r]}"
            else:
                ann[r, c] = f"{wins[r, c]}/{non_ties[r, c]}"

    # подписи
    def _lab(cid: int) -> str:
        if labels and cid in labels:
            return labels[cid]
        return str(cid)

    tick_labels = [_lab(cid) for cid in order]

    # === Рисуем ===
    fig, ax = plt.subplots(figsize=figsize)

    # теплокарта только вне диагонали
    data = np.ma.masked_where(mask_diag, share)
    im = ax.imshow(data, vmin=0.0, vmax=1.0, cmap="RdYlGn")

    # диагональ закрасим нейтральным серым
    for i in range(m):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, facecolor="lightgray", edgecolor="none", zorder=2))

    # сетка
    ax.set_xticks(np.arange(m))
    ax.set_yticks(np.arange(m))
    
    ax.set_xticklabels(tick_labels)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticklabels(tick_labels)

    ax.set_xticks(np.arange(-.5, m, 1), minor=True)
    ax.set_yticks(np.arange(-.5, m, 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # аннотации текста
    for r in range(m):
        for c in range(m):
            # чуть меньше шрифт, чтобы влезло
            ax.text(c, r, ann[r, c], ha="center", va="center", fontsize=8, zorder=3)

    # colorbar и заголовок
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Доля побед (wins / non-ties)")

    ax.set_title(f"IRV pairwise matrix — всего рейтингов: {n_ballots}")

    ax.set_xlabel("Сравниваемый кандидат (столбец)")
    ax.set_ylabel("Кандидат в строке")

    fig.tight_layout()
    return fig, ax
