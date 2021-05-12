import rouge
import argparse
import codecs


def prepare_results(metric, p, r, f):
    return '\t{}:\t {:5.2f}\t {:5.2f}\t {:5.2f}'.format(metric, 100.0 * p, 100.0 * r,
                                                                 100.0 * f)


def test_rouge(candidates, references):
    candidates = [line.strip() for line in candidates]
    references = [line.strip() for line in references]
    assert len(candidates) == len(references)

    apply_avg = True
    apply_best = False

    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                            max_n=4,
                            limit_length=True,
                            length_limit=100,
                            length_limit_type='words',
                            apply_avg=apply_avg,
                            apply_best=apply_best,
                            alpha=0.5,  # Default F1_score
                            weight_factor=1.2,
                            stemming=True)

    all_hypothesis = candidates
    all_references = references

    scores = evaluator.get_scores(all_hypothesis, all_references)

    rougel = ""
    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
        if metric in ["rouge-1", "rouge-2", "rouge-l"]:
            print(prepare_results(metric, results['p'], results['r'], results['f']))
            rougel = rougel + '{:5.2f}'.format(100 * results['f']) + "-"

    print("ROUGE 1-2-L F:", rougel)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str, default="candidate.txt",
                        help='candidate file')
    parser.add_argument('-r', type=str, default="data/golden.txt",
                        help='reference file')
    args = parser.parse_args()

    candidates = codecs.open(args.c, encoding="utf-8")
    references = codecs.open(args.r, encoding="utf-8")

    test_rouge(candidates, references)
