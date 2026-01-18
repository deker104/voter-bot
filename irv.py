from typing import List, Tuple

# =========================
# IRV подсчёт (опционально)
# =========================

def irv_tally(ballots: List[List[int]], candidate_ids: List[int]) -> Tuple[int, List[dict]]:
    """
    Возвращает (winner_id, rounds)
    rounds: список словарей {round, counts, exhausted, eliminated?}
    """
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

        round_info = {
            "round": rnd,
            "counts": dict(sorted(counts.items(), key=lambda x: (-x[1], x[0]))),
            "exhausted": exhausted,
            "total_active_votes": total_active_votes,
        }

        # Если остался 1 кандидат
        if len(active) == 1:
            winner = next(iter(active))
            rounds.append(round_info)
            return winner, rounds

        # Победа по большинству (строго > 50% от активных)
        if total_active_votes > 0:
            for cid, v in counts.items():
                if v > total_active_votes / 2:
                    rounds.append(round_info)
                    return cid, rounds

        # Иначе устраняем кандидата с минимумом голосов.
        # Тайбрейк: минимальный count, затем минимальный id.
        min_votes = min(counts.values()) if counts else 0
        losers = [cid for cid, v in counts.items() if v == min_votes]
        eliminated = min(losers)  # стабильный тайбрейк

        round_info["eliminated"] = eliminated
        rounds.append(round_info)

        active.remove(eliminated)
        rnd += 1
