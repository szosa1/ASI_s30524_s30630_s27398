# Model Card

> Scaffold for Sprint 3 — do not implement before this sprint.
> Minimalną Model Card wypełnia zespół w Sprincie 3, a aktualizuje w Sprincie 6.

## Model

- Typ modelu i jego wersja (alias `production` w W&B): Regresja logistyczna (`baseline_logreg`), wersja v0 alias `production`.
- Data utworzenia i autor runu: 26.11.2026 autor: s30524

## Przeznaczenie

- Problem (binary classification) i klasa pozytywna: Przewidywanie, czy użytkownik sklepu internetowego dokona zakupu.
Klasa pozytywna to `Revenue = True`.
- Użytkownik systemu i decyzja, którą wspiera model: Właściciel lub analityk sklepu internetowego. Model wspiera analize zachowań klientów i ocene prawdopodobieństwa zakupu.
- Zastosowania, do których model nie jest przeznaczony: Podejmowanie innych automatycznych decyzji o klientach bez dodatkowej weryfikacji.

## Dane

- Dataset (id z `docs/datasets.md`) i fingerprint danych: `online_shoppers`. SHA-256 pliku źródłowego: `b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9`.
- Strategia splitu z kontraktu i liczebności zbiorów: `stratified_random`, 80% danych treningowych (9864 rekordy) i 20% testowych (2466 rekordów), `random_state=42`.

## Metryki

- Metryka główna na zbiorze testowym, w porównaniu z DummyClassifier: F1 = 0.3852 dla `baseline_logreg`, a dla `DummyClassifier` F1 = 0.0.
- Pozostałe metryki i macierz pomyłek: accuracy = 0.6298, precision = 0.2593, recall = 0.7487. Macierz pomyłek: TN = 1267, FP = 817, FN = 96, TP = 286.

## Ograniczenia i ryzyka

- Ograniczenia danych (z `docs/datasets.md`): Tylko 15,5% wizyt kończy się zakupem. Niektóre informacje o wizytach są trudne do zebrania.
- Znane błędy i sytuacje, w których model zawodzi: Model błędnie przewidział zakup 817 razy i nie wykrył 96 zakupów. Nie wiemy też jak dobrze działałby w np. innych lub specjalistycznych sklepach.

## Wersje

- Wersje bibliotek modelu i git SHA treningu: Python 3.11.18, scikit-learn 1.7.2, pandas 2.3.3, numpy 2.2.6, joblib 1.6.0. Git SHA: `6a5b459d6e12e5077bc6bbf0115ebb72a15e03d9`.
