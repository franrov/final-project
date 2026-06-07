# Final Assignment — Written Analysis

Character-level generative Transformer (miniature GPT), implemented in
[`src/models.py`](src/models.py) (Task 1) and [`src/gpt_model.py`](src/gpt_model.py)
(Task 2), trained with [`src/train_gpt.py`](src/train_gpt.py).

---

## 1. Corpus

**Corpus:** Shakespeare (stated in README)

**Why this corpus.** Three reasons:

1. **Strong, well-understood structure.** A Shakespeare play has a very regular
   surface form — an ALL-CAPS speaker name, a colon, a newline, then dialogue.
   That structure is learnable from raw characters alone and gives a clear,
   visual signal of whether the model is learning: if the samples start
   producing `NAME:` headers followed by line breaks, the model has captured
   long-range layout.
2. **Vocabulary.** Only 65 distinct characters, so the character-level vocabulary is small and the
   model can be trained on a CPU, while the language itself is varied enough to
   force the model to learn spelling, word boundaries, and things like punctuation.
3. **Reproducability.**  Loss values and sample quality can be sanity-checked against widely
   published results for a model of this size.

---

## 2. Results

### 2a. Baseline run

Config: `BLOCK_SIZE=64, LAYER_SIZE=64, N_LAYERS=2, LR=3e-4, MAX_ITERS=3000,
BATCH_SIZE=32` — **41,921 parameters**. Full log:
[`results/train_default.log`](results/train_default.log).

Last 10 rows of the loss table:

| Step | Train Loss | Val Loss |
|-----:|-----------:|---------:|
|  300 | 2.5734 | 2.5749 |
|  600 | 2.4632 | 2.4765 |
|  900 | 2.4097 | 2.4363 |
| 1200 | 2.3889 | 2.4059 |
| 1500 | 2.3673 | 2.3889 |
| 1800 | 2.3584 | 2.3801 |
| 2100 | 2.3478 | 2.3682 |
| 2400 | 2.3321 | 2.3459 |
| 2700 | 2.3185 | 2.3446 |
| 2999 | **2.3160** | **2.3339** |

Generated sample (500 chars, seeded with the first character of the corpus):

```
Fre: mud; to so Ditoor's psind s, weriougerd s ay'st he ad lif nery trelinchiveses?
Mupse.


TAUMICSINBO, wind Gyock ESIO:
Girg.
MIANG I soghers yO:
Whe wayore I Gage 'lad winciet t iss be whead peerg,
whan'e co, perofpl fo, he ifotr
Toreno had we no?
Benever tsire mas merd cavath wor ferunoo; teed t e, wan wo cendotoure thas re thay a mior
Bue heean sene st then thandain mo d!

RERMunties haghy alare I wesit weche b testheld baue Re!, aid st E thinghist'de
Juchil illfours by imin dis;
Ther mure
```

**What the model learned.** Even at 42k parameters the model captured the
*format* of a play — capitalized speaker tags ending in a colon
(`TAUMICSINBO`, `ESIO:`, `MIANG`), blank lines between speeches, and
plausible word lengths with sensible vowel/consonant alternation. It has not
learned real English words.

### 2b. Larger configuration (the required experiment)

Config: `LAYER_SIZE=128, N_LAYERS=4, MAX_ITERS=4000` (block size and learning
rate unchanged) — **281,665 parameters** (~6.7× larger). Full log:
[`results/train_large.log`](results/train_large.log).

Last 10 rows of the loss table:

| Step | Train Loss | Val Loss |
|-----:|-----------:|---------:|
| 1200 | 2.2146 | 2.2566 |
| 1500 | 2.1609 | 2.2141 |
| 1800 | 2.1149 | 2.1702 |
| 2100 | 2.0731 | 2.1455 |
| 2400 | 2.0366 | 2.1300 |
| 2700 | 2.0122 | 2.1049 |
| 3000 | 1.9806 | 2.0683 |
| 3300 | 1.9488 | 2.0624 |
| 3600 | 1.9269 | 2.0207 |
| 3999 | **1.8868** | **1.9988** |

Generated sample (500 chars):

```
Fory, stweer dikesputs: my poer hangurpy.

CIChave all worn's not thy; witere.

BELY:
St Virdilden:
Tho beamer, an hen you core beang firt.

BY ENAUS:
I ma all. fortul wnothy men is slen, I't'll melve hupice?

ENIUS:
Soly; be good ind son ayalll ife, yepirent for yours le.
I frothm bee word that this suwe: I
I that doney'd wil-pod strool she wifity, loke not
Thin loven, at.

Fa ethe tollfo The Comeest Fonce severfeie sunnt.-stelanty Edo Herue:
Shis and that one sall.
```

### 2c. Commentary — effect of larger `N_LAYERS` / `LAYER_SIZE` / `MAX_ITERS`

| Metric | Baseline (42k) | Larger (282k) | Change |
|---|---|---|---|
| Final train loss | 2.3160 | 1.8868 | −0.43 |
| Final val loss | 2.3339 | 1.9988 | −0.34 |

Observations:

- **Lower loss across the board.** Widening the hidden dimension (64→128) and
  stacking more blocks (2→4) gave the model more capacity to represent
  character dependencies, and the extra 1000 steps let it use that capacity.
  Validation loss fell from 2.33 to 2.00.
- **Visibly better text.** The larger model produces actual English words —
  `good`, `men`, `word`, `that`, `this`, `all`, `not`, `son` — and uses
  apostrophes in word-like contractions (`I'll`, `doney'd`). The play layout is
  also cleaner, with multiple well-formed `NAME:` headers.
- **It is still a small model.** Both runs overfit only mildly (train and val loss
  track closely), which means we were **capacity-limit, not exactly data-limit** — therefore,
  the gap between val loss ~2.0 and fluent English shows that further gains
  would come from yet more layers/width and longer training, not more information.
- **Diminishing returns/cost.** The larger run took meaningfully longer for a
  ~0.34 val-loss gain. But, I understand that, on a CPU this is the trade-off: each doubling of
  width roughly quadruples the per-step compute in the linear layers.
