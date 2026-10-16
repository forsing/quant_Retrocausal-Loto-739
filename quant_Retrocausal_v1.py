#!/usr/bin/env python3


"""
Retrocausal V1 — dvosmerno ucenje za Loto 7/39.

Smer unapred:
    prethodna dva stanja -> naredno stanje

Povratna rekonstrukcija:
    kandidat narednog stanja + starije poznato stanje
    -> rekonstrukcija poslednjeg poznatog stanja

Buduci rezultati se ne koriste.
Skor kandidata nije verovatnoca dobitka.
"""


import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np


DEFAULTS = [
    "/Users/4c/Desktop/GHQ/data/loto7_4702_k82.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4702_k82_loto_2973.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4702_k82_loto_plus_1729.csv",
]

TRAIN = 192
VALIDATION = 48
TEST = 64
REFIT = 16
ANCHORS = 32

PENALTIES = (0.03, 0.30)
REVERSE_WEIGHTS = (0.0, 0.1, 0.5, 2.0)

PAIR = np.asarray(
    list(combinations(range(7), 2)),
    dtype=np.int32,
)


def read_csv(path):
    result = []

    with Path(path).open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for line, cells in enumerate(csv.reader(handle), 1):
            if not cells or all(not x.strip() for x in cells):
                continue

            try:
                row = tuple(
                    sorted(int(x.strip()) for x in cells)
                )
            except ValueError as exc:
                raise ValueError(
                    f"{path}, red {line}: ocekujem cele brojeve."
                ) from exc

            if (
                len(row) != 7
                or len(set(row)) != 7
                or not 1 <= row[0] <= row[-1] <= 39
            ):
                raise ValueError(
                    f"{path}, red {line}: potrebno je "
                    "7 razlicitih brojeva 1..39."
                )

            result.append(row)

    minimum = TRAIN + VALIDATION + TEST + 2

    if len(result) < minimum:
        raise ValueError(
            f"{path}: potrebno najmanje {minimum} kola."
        )

    return np.asarray(result, dtype=np.int32)


def profile(rows):
    """
    Opis cele kombinacije i distribucije njenih
    21 parnih rastojanja.

    Nema istorijskog frekvencijskog rangiranja brojeva.
    """
    a = np.atleast_2d(rows).astype(np.float64)

    distance = (
        a[:, PAIR[:, 1]]
        - a[:, PAIR[:, 0]]
    )

    cdf = np.stack(
        [
            (distance <= boundary).mean(axis=1)
            for boundary in (2, 5, 10, 20)
        ],
        axis=1,
    )

    span = (
        a[:, -1:] - a[:, :1]
    ) / 38.0

    return np.concatenate(
        (a / 39.0, cdf, span),
        axis=1,
    )


def rational_kernel(a, b, bandwidth):
    """Racionalna slicnost, bez normalne raspodele."""
    distance = np.maximum(
        (a * a).sum(axis=1)[:, None]
        + (b * b).sum(axis=1)[None, :]
        - 2 * a @ b.T,
        0.0,
    ) / a.shape[1]

    return 1.0 / (
        1.0 + distance / bandwidth
    )


class Ridge:
    def __init__(self, x, y, penalty):
        self.xmean = x.mean(axis=0)

        self.xscale = np.maximum(
            x.std(axis=0),
            1e-6,
        )

        self.ymean = y.mean(axis=0)

        z = (
            x - self.xmean
        ) / self.xscale

        gram = z.T @ z / len(z)

        rhs = (
            z.T @ (y - self.ymean)
        ) / len(z)

        self.coef = np.linalg.solve(
            gram + penalty * np.eye(z.shape[1]),
            rhs,
        )

    def predict(self, x):
        z = (
            x - self.xmean
        ) / self.xscale

        return self.ymean + z @ self.coef


class BidirectionalModel:
    def __init__(self, history, penalty):
        # history sadrzi samo prethodno poznata kola.
        start = max(
            2,
            len(history) - TRAIN,
        )

        features = profile(history)
        fit = features[start - 2:]

        self.center = np.median(
            fit,
            axis=0,
        )

        q25, q75 = np.quantile(
            fit,
            [0.25, 0.75],
            axis=0,
        )

        self.scale = np.maximum(
            q75 - q25,
            0.03,
        )

        standardized = (
            features - self.center
        ) / self.scale

        # Deterministicki izbor referentnih stanja.
        ids = np.linspace(
            start - 2,
            len(history) - 1,
            ANCHORS,
        ).astype(int)

        self.anchors = standardized[ids]

        distances = (
            (
                self.anchors[:, None]
                - self.anchors[None, :]
            ) ** 2
        ).mean(axis=2)

        positive = distances[
            np.triu_indices(ANCHORS, 1)
        ]

        positive = positive[
            positive > 1e-12
        ]

        self.bandwidth = (
            float(np.median(positive))
            if len(positive)
            else 1.0
        )

        q = rational_kernel(
            standardized,
            self.anchors,
            self.bandwidth,
        )

        # Istorijska trojka:
        # older = t-2
        # current = t-1
        # following = t
        older = q[start - 2:-2]
        current = q[start - 1:-1]
        following = q[start:]

        self.forward = Ridge(
            np.concatenate(
                (current, older),
                axis=1,
            ),
            following,
            penalty,
        )

        self.reverse = Ridge(
            np.concatenate(
                (following, older),
                axis=1,
            ),
            current,
            penalty,
        )

        self.forward_scale = np.maximum(
            following.std(axis=0),
            0.02,
        )

        self.reverse_scale = np.maximum(
            current.std(axis=0),
            0.02,
        )

    def embed(self, rows):
        standardized = (
            profile(rows) - self.center
        ) / self.scale

        return rational_kernel(
            standardized,
            self.anchors,
            self.bandwidth,
        )

    def predict(self, history, reverse_weight):
        older, current = self.embed(
            history[-2:]
        )

        forward_target = self.forward.predict(
            np.concatenate(
                (current, older)
            )
        )

        def costs(candidates):
            q = self.embed(candidates)

            result = (
                (
                    (q - forward_target)
                    / self.forward_scale
                ) ** 2
            ).mean(axis=1)

            if reverse_weight > 0:
                past = np.broadcast_to(
                    older,
                    q.shape,
                )

                reconstructed = self.reverse.predict(
                    np.concatenate(
                        (q, past),
                        axis=1,
                    )
                )

                result += reverse_weight * (
                    (
                        (reconstructed - current)
                        / self.reverse_scale
                    ) ** 2
                ).mean(axis=1)

            if not np.isfinite(result).all():
                raise RuntimeError(
                    "Neispravan skor modela."
                )

            return result

        # Pocetne tacke pretrage nisu obavezni izlazi.
        recent = history[-96:]

        order = np.argsort(
            costs(recent),
            kind="stable",
        )[:12]

        seeds = {
            tuple(map(int, recent[i]))
            for i in order
        }

        seeds.update(
            tuple(map(int, row))
            for row in history[-8:]
        )

        def rank(candidates):
            candidates = sorted(candidates)

            values = costs(
                np.asarray(
                    candidates,
                    dtype=np.int32,
                )
            )

            order = sorted(
                range(len(candidates)),
                key=lambda i: (
                    float(values[i]),
                    candidates[i],
                ),
            )

            return [
                candidates[i]
                for i in order[:4]
            ]

        beam = rank(seeds)

        best = float(
            costs(
                np.asarray(beam[:1])
            )[0]
        )

        # Deterministicka pretraga zamenom jednog broja.
        # Ogranicenja su samo pravila igre.
        for _ in range(6):
            candidates = set(beam)

            for row in beam:
                used = set(row)

                for removed in row:
                    others = used - {removed}

                    for added in range(1, 40):
                        if added not in used:
                            candidates.add(
                                tuple(
                                    sorted(
                                        others | {added}
                                    )
                                )
                            )

            new_beam = rank(candidates)

            value = float(
                costs(
                    np.asarray(new_beam[:1])
                )[0]
            )

            if value >= best - 1e-12:
                break

            beam = new_beam
            best = value

        result = beam[0]

        assert len(result) == len(set(result)) == 7
        assert 1 <= result[0] <= result[-1] <= 39

        return result


def evaluate(
    rows,
    start,
    stop,
    penalty,
    reverse_weight,
):
    predictions = []
    matches = []

    for t in range(start, stop):
        if (t - start) % REFIT == 0:
            model = BidirectionalModel(
                rows[:t],
                penalty,
            )

        prediction = model.predict(
            rows[:t],
            reverse_weight,
        )

        predictions.append(prediction)

        matches.append(
            len(
                set(prediction)
                & set(map(int, rows[t]))
            )
        )

    return (
        np.asarray(matches, dtype=np.int32),
        predictions,
    )


def calibrate(rows, start, stop):
    records = []

    for penalty in PENALTIES:
        for weight in REVERSE_WEIGHTS:
            values, _ = evaluate(
                rows,
                start,
                stop,
                penalty,
                weight,
            )

            mean = float(values.mean())

            records.append(
                (mean, penalty, weight)
            )

            print(
                f"Validacija ridge={penalty:g}, "
                f"povratno={weight:g}: {mean:.4f}",
                flush=True,
            )

    # Kod jednakog rezultata prednost imaju
    # manja povratna tezina i veca regularizacija.
    # Test period ne utice na izbor.
    best = max(
        records,
        key=lambda record: (
            record[0],
            -record[2],
            record[1],
        ),
    )

    forward = max(
        (
            record
            for record in records
            if record[2] == 0
        ),
        key=lambda record: (
            record[0],
            record[1],
        ),
    )

    return best[1:], forward[1:]


def longest_run(row):
    best = current = 1

    for a, b in zip(row, row[1:]):
        current = (
            current + 1
            if b == a + 1
            else 1
        )

        best = max(best, current)

    return best


def self_check():
    rows = np.asarray(
        [
            sorted({
                (3 * i + 5 * j) % 39 + 1
                for j in range(7)
            })
            for i in range(240)
        ],
        dtype=np.int32,
    )

    model = BidirectionalModel(
        rows[:210],
        PENALTIES[0],
    )

    altered = rows.copy()
    altered[210:] = np.arange(1, 8)

    other = BidirectionalModel(
        altered[:210],
        PENALTIES[0],
    )

    assert np.array_equal(
        model.anchors,
        other.anchors,
    )

    assert np.array_equal(
        model.forward.coef,
        other.forward.coef,
    )

    assert np.array_equal(
        model.reverse.coef,
        other.reverse.coef,
    )

    for weight in (0.0, 0.5):
        first = model.predict(
            rows[:210],
            weight,
        )

        assert first == model.predict(
            rows[:210],
            weight,
        )

        assert first == other.predict(
            altered[:210],
            weight,
        )


def run(path):
    rows = read_csv(path)
    n = len(rows)

    test_start = n - TEST
    validation_start = (
        test_start - VALIDATION
    )

    print(
        f"\n{Path(path).name}: {n} kola",
        flush=True,
    )

    selected, forward_config = calibrate(
        rows,
        validation_start,
        test_start,
    )

    values, predictions = evaluate(
        rows,
        test_start,
        n,
        *selected,
    )

    if selected == forward_config:
        forward_values = values.copy()
    else:
        forward_values, _ = evaluate(
            rows,
            test_start,
            n,
            *forward_config,
        )

    # Kontrola sa istom regularizacijom izdvaja
    # doprinos same povratne provere.
    if selected[0] == forward_config[0]:
        matched_values = forward_values
    else:
        matched_values, _ = evaluate(
            rows,
            test_start,
            n,
            selected[0],
            0.0,
        )

    previous = np.asarray([
        len(
            set(map(int, rows[t - 1]))
            & set(map(int, rows[t]))
        )
        for t in range(test_start, n)
    ])

    historic = sum(
        any(
            tuple(map(int, row)) == prediction
            for row in rows[:t]
        )
        for t, prediction in zip(
            range(test_start, n),
            predictions,
        )
    )

    repeated_last = sum(
        prediction
        == tuple(map(int, rows[t - 1]))
        for t, prediction in zip(
            range(test_start, n),
            predictions,
        )
    )

    print(
        f"Izabrano: ridge={selected[0]:g}, "
        f"povratna tezina={selected[1]:g}"
    )

    print(
        "Kontrola samo unapred: "
        f"ridge={forward_config[0]:g}"
    )

    print(
        f"TEST V1: {values.mean():.4f}"
    )

    print(
        "TEST samo unapred, isti ridge: "
        f"{matched_values.mean():.4f}"
    )

    print(
        "TEST samo unapred, zasebna validacija: "
        f"{forward_values.mean():.4f}"
    )

    print(
        "Kontrola: prethodno izvuceno kolo: "
        f"{previous.mean():.4f}"
    )

    print(
        "Pogodaka 0..7:",
        np.bincount(
            values,
            minlength=8,
        ).tolist(),
    )

    print(
        f"3+: {int((values >= 3).sum())}/{TEST}; "
        f"4+: {int((values >= 4).sum())}/{TEST}"
    )

    print(
        f"Razlicitih predloga: "
        f"{len(set(predictions))}/{TEST}; "
        f"istorijskih: {historic}/{TEST}"
    )

    print(
        f"Kopija poslednjeg kola: "
        f"{repeated_last}/{TEST}"
    )

    print(
        "Predloga sa nizom 5+:",
        sum(
            longest_run(prediction) >= 5
            for prediction in predictions
        ),
    )

    print(
        "Predloga potpuno u jednoj polovini:",
        sum(
            prediction[-1] <= 19
            or prediction[0] >= 20
            for prediction in predictions
        ),
    )

    model = BidirectionalModel(
        rows,
        selected[0],
    )

    prediction = model.predict(
        rows,
        selected[1],
    )

    assert prediction == model.predict(
        rows,
        selected[1],
    )

    print(
        f"NEXT red {n + 1}: "
        + ",".join(map(str, prediction))
    )

    print(
        "Postoji u istoriji:",
        any(
            tuple(map(int, row)) == prediction
            for row in rows
        ),
    )

    print(
        f"Najduzi niz: {longest_run(prediction)}; "
        f"1..19: {sum(x <= 19 for x in prediction)}"
    )

    if selected[1] == 0:
        print(
            "Validacija je iskljucila povratnu proveru; "
            "NEXT koristi smer unapred."
        )
    elif values.mean() <= matched_values.mean():
        print(
            "Povratna provera nije poboljsala "
            "rezultat pri istoj regularizaciji."
        )

    if values.mean() <= previous.mean():
        print(
            "V1 nije nadmasila kontrolu "
            "prethodnog kola."
        )

    return prediction


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Retrocausal V1 — dvosmerno ucenje "
            "za Loto 7/39."
        )
    )

    parser.add_argument(
        "csv",
        nargs="*",
        help="CSV bez zaglavlja: 7 brojeva po redu.",
    )

    args = parser.parse_args()

    self_check()

    print(
        "Retrocausal V1: kandidati se proveravaju "
        "unapred i unazad."
    )

    print(
        "Buduci rezultati se ne koriste. "
        "Skor nije verovatnoca dobitka.",
        flush=True,
    )

    results = [
        (path, run(path))
        for path in (args.csv or DEFAULTS)
    ]

    # Isti izlaz se prijavljuje, bez vestackog menjanja.
    for i, (path, combo) in enumerate(results):
        for other_path, other in results[:i]:
            if combo == other:
                print(
                    f"Isti NEXT: {Path(path).name} / "
                    f"{Path(other_path).name}: {combo}"
                )


if __name__ == "__main__":
    main()



"""
Retrocausal V1: kandidati se proveravaju unapred i unazad.
Buduci rezultati se ne koriste. Skor nije verovatnoca dobitka.


loto7_4702_k82.csv: 4702 kola
Validacija ridge=0.03, povratno=0: 1.2500
Validacija ridge=0.03, povratno=0.1: 1.3958
Validacija ridge=0.03, povratno=0.5: 1.2500
Validacija ridge=0.03, povratno=2: 1.2500
Validacija ridge=0.3, povratno=0: 1.2500
Validacija ridge=0.3, povratno=0.1: 1.2292
Validacija ridge=0.3, povratno=0.5: 1.2500
Validacija ridge=0.3, povratno=2: 1.2083
Izabrano: ridge=0.03, povratna tezina=0.1
Kontrola samo unapred: ridge=0.3
TEST V1: 1.3594
TEST samo unapred, isti ridge: 1.4375
TEST samo unapred, zasebna validacija: 1.5469
Kontrola: prethodno izvuceno kolo: 1.1719
Pogodaka 0..7: [10, 32, 12, 9, 1, 0, 0, 0]
3+: 10/64; 4+: 1/64
Razlicitih predloga: 64/64; istorijskih: 0/64
Kopija poslednjeg kola: 0/64
Predloga sa nizom 5+: 0
Predloga potpuno u jednoj polovini: 0
NEXT red 4703: 7,10,21,22,25,28,31
Postoji u istoriji: False
Najduzi niz: 2; 1..19: 2
Povratna provera nije poboljsala rezultat pri istoj regularizaciji.



loto7_4702_k82_loto_2973.csv: 2973 kola
Validacija ridge=0.03, povratno=0: 0.9375
Validacija ridge=0.03, povratno=0.1: 0.9583
Validacija ridge=0.03, povratno=0.5: 1.1250
Validacija ridge=0.03, povratno=2: 1.2292
Validacija ridge=0.3, povratno=0: 1.2917
Validacija ridge=0.3, povratno=0.1: 1.4167
Validacija ridge=0.3, povratno=0.5: 1.2500
Validacija ridge=0.3, povratno=2: 1.2083
Izabrano: ridge=0.3, povratna tezina=0.1
Kontrola samo unapred: ridge=0.3
TEST V1: 1.3750
TEST samo unapred, isti ridge: 1.2500
TEST samo unapred, zasebna validacija: 1.2500
Kontrola: prethodno izvuceno kolo: 1.3281
Pogodaka 0..7: [14, 18, 26, 6, 0, 0, 0, 0]
3+: 6/64; 4+: 0/64
Razlicitih predloga: 63/64; istorijskih: 0/64
Kopija poslednjeg kola: 0/64
Predloga sa nizom 5+: 0
Predloga potpuno u jednoj polovini: 0
NEXT red 2974: 5,6,9,22,25,26,37
Postoji u istoriji: False
Najduzi niz: 2; 1..19: 3



loto7_4702_k82_loto_plus_1729.csv: 1729 kola
Validacija ridge=0.03, povratno=0: 1.0833
Validacija ridge=0.03, povratno=0.1: 1.1042
Validacija ridge=0.03, povratno=0.5: 1.0625
Validacija ridge=0.03, povratno=2: 1.2083
Validacija ridge=0.3, povratno=0: 1.0833
Validacija ridge=0.3, povratno=0.1: 1.0417
Validacija ridge=0.3, povratno=0.5: 1.2292
Validacija ridge=0.3, povratno=2: 1.0625
Izabrano: ridge=0.3, povratna tezina=0.5
Kontrola samo unapred: ridge=0.3
TEST V1: 1.1875
TEST samo unapred, isti ridge: 1.2500
TEST samo unapred, zasebna validacija: 1.2500
Kontrola: prethodno izvuceno kolo: 1.2656
Pogodaka 0..7: [18, 23, 16, 7, 0, 0, 0, 0]
3+: 7/64; 4+: 0/64
Razlicitih predloga: 64/64; istorijskih: 0/64
Kopija poslednjeg kola: 0/64
Predloga sa nizom 5+: 0
Predloga potpuno u jednoj polovini: 0
NEXT red 1730: 4,5,11,12,32,33,34
Postoji u istoriji: False
Najduzi niz: 3; 1..19: 4
Povratna provera nije poboljsala rezultat pri istoj regularizaciji.
V1 nije nadmasila kontrolu prethodnog kola.
"""



"""
retrokauzalnost — i vremenski simetrični opisi
dvosmerno učenje prelaza 

1. Smer napred: iz prethodnih kombinacija uči opis distribucije naredne kombinacije.
2. Smer nazad: na poznatim istorijskim prelazima uči rekonstrukciju prethodnog stanja iz narednog.
3. Pri izboru NEXT: kandidat se ocenjuje i prema predikciji unapred i prema tome koliko njegova povratna rekonstrukcija odgovara stvarno poznatoj prošlosti.

Buduća kombinacija ovde je kandidat koji proveravam. 
Nijedan nepoznati rezultat ne ulazi u učenje ili proveru.
Za razliku od Diraka, gde povezujem stanje i promenu, ovde proveravam saglasnost prelaza u oba smera.
Povratni model može samo da favorizuje ponavljanje poslednjeg stanja. 
Zato njegov doprinos treba zasebno uporediti sa modelom koji predviđa samo unapred.


Poslednja 64 kola, nakon izbora parametara na prethodnih 48:

CSV	            Dvosmerni V1	Samo unapred, ista regularizacija	Prethodno kolo
Zajednički 4702	1,3594	        1,4375	                            1,1719
Loto 2973	    1,3750	        1,2500	                            1,3281
Plus 1729	    1,1875	        1,2500	                            1,2656

Povratna provera poboljšala je rezultat samo na Loto CSV-u. 
Nema random postupaka ni ručnih filtera oblika. 
U testu nije bilo nizova od pet uzastopnih brojeva, kombinacija potpuno u jednoj polovini ili kopiranja poslednjeg kola.
"""
