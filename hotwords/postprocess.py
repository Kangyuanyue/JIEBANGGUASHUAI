import editdistance

def match_hotword(text, hotwords, threshold=0.45):
    if not text:
        return text, None, 0.0

    best_match = None
    best_ratio = float('inf')

    for hw in hotwords:
        dist = editdistance.eval(text, hw)
        ratio = dist / max(len(text), len(hw))
        if ratio < best_ratio:
            best_ratio = ratio
            best_match = hw

    if best_ratio <= threshold:
        return best_match, best_match, best_ratio
    else:
        return text, None, best_ratio