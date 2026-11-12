# ADR-1: Gdzie żyje przetwarzanie danych

- **Status:** Przyjęty
- **Sprint:** S2
- **Autor:** Igor Kordaczuk @PyszneMaselko
- **Recenzent:** Jakub Jaworski @jamejj

## Kontekst

Przenosimy trening z notebooka do Kedro (`data_processing` -> `data_science`). W dalszej części projektu
ten sam model będzie obsługiwał pojedyncze rekordy w API. Musimy ustalić, gdzie żyje każdy
z trzech rodzajów przetwarzania, żeby trening i predykcja widziały dane w tej samej postaci
i żeby żadna statystyka z danych testowych nie przeciekła do modelu.

Dane: UCI Online Shoppers Purchasing Intention, 12 330 sesji, target `Revenue`

## Rozważone opcje

1. **Podział na trzy warstwy (wybrana):** input normalization w `data.py` (wspólne z API),
   training-only filtering jako osobny node Kedro, model preprocessing wewnątrz sklearn `Pipeline`.
2. **Całe przetwarzanie w nodach Kedro:** imputacja, skalowanie i one-hot jako osobne nody
   przed treningiem, a model to sam `LogisticRegression`.

## Decyzja

Wybieramy opcję 1:

| Rodzaj | Gdzie żyje | Kiedy działa | Dlaczego tam |
|---|---|---|---|
| Input normalization | `src/asi_project/data.py::normalize_input`, node `normalize_input_node` w `data_processing` | trening i API | działa na pojedynczym rekordzie, nie używa targetu ani statystyk zbioru, jest idempotentna - można ją bezpiecznie wywołać na żądaniu z API; jedna implementacja = brak rozjazdu trening/serving |
| Training-only filtering | `data.py::filter_training_rows`, node `filter_training_rows_node` | tylko przy przygotowaniu danych treningowych | usuwa rekordy np. bez targetu w API nie ma targetu, a odrzucenie żądania to inna decyzja niż odfiltrowanie wiersza treningowego |
| Model preprocessing | `src/asi_project/model.py::build_baseline_pipeline` (`ColumnTransformer`: mediana + `StandardScaler`, most_frequent + `OneHotEncoder(handle_unknown="ignore")`) | wewnątrz modelu, `fit` tylko na `X_train` | statystyki (mediany, skale, kategorie) są liczone wyłącznie na zbiorze treningowym i zapisywane razem z modelem w jednym artefakcie (`data/06_models/model.pkl`), więc API wykona dokładnie te same kroki |

Którą opcję wybieramy i dlaczego. Co najmniej dwie liczby z naszego systemu, na których
opiera się decyzja:

| Wielkość | Wartość | Skąd pochodzi |
|---|---|---|
| Rekordy w surowych danych | 12 330 | `online_shoppers_intention.csv` |
| Rekordy bez targetu usunięte przez filtering | 0 | `filter_training_rows`, notebook 01 |
| Dokładne duplikaty (świadomie **nie** usuwane) | 125 | notebook 01 - mogą to być różne sesje o identycznych cechach |
| Split train / test | 9 864 / 2 466 (`test_size=0.2`, `random_state=42`, stratified) | `conf/base/parameters.yml`, `metrics.json` |
| F1 LogisticRegression vs Dummy (pełne dane) | 0,385 vs 0,000 | `metrics.json` |

## Odrzucone alternatywy

**Opcja 2 (Całe przetwarzanie w nodach Kedro):** odrzucona, bo dopasowane imputery, skalery
  i encodery trzeba by zapisywać i wczytywać w API osobno od modelu. Łatwo wtedy o rozjazd
  wersji albo o `fit` na całym zbiorze przed splitem (wyciek danych testowych). Artefakt w postaci
  jednego sklearn `Pipeline` tego problemu nie ma.

## Konsekwencje

- **Ułatwienie 1:** API (S4) importuje `normalize_input` z `data.py` i ładuje jeden artefakt modelu.
  Nie ma drugiej kopii logiki ani osobnych plików z dopasowanymi transformacjami.
- **Ułatwienie 2:** nowe reguły normalizacji testujemy raz (`tests/test_project.py`: idempotencja, brak zależności od innych rekordów).

## Historia

| Data | Zmiana |
|---|---|
| 2026-11-12 | utworzenie |
