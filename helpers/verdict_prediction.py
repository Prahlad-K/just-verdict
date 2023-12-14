
def get_verdict(entailment_scores, low=-0.6599999999999997, high=-0.6299999999999997):
    index_values = [-1, 1, 0]
    verdict = 0

    for subclaim_entailment_scores in entailment_scores:
        which_verdict = subclaim_entailment_scores.argmax()
        verdict += index_values[which_verdict]*subclaim_entailment_scores[which_verdict]

    if verdict > high:
        return 1
    elif verdict > low:
        return 0
    else:
        return -1