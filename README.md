# Hidden Markov Models

## From the lecture to code

In Lecture 5 you saw how an HMM explains a sequence by a hidden path, and how dynamic
programming answers the questions *which path is most likely?* (Viterbi) and *which
state is most likely at position i?* (Forward, Backward, posterior). Today you
implement them yourselves.

Structure of the session:
1. **The model in code** (theory) → **Task 1**
2. **Viterbi** (theory) → **Task 2**
3. **Forward, Backward, posterior** (theory) → **Task 3**

The paper parts of the tasks (hand computations on a small toy, and the questions that
go with them) are on the printed handout. This README has the code.

**Where your answers go.** Write the answers to the paper tasks and to the handout's
*Think about it* questions on the handout. Write your own code from scratch in a single
file, `hmm_exercise.py`, and answer the *Think about it* questions of this README in a
comment at the end of that file. Fork this repository to your own GitHub account, clone
your fork, and commit and push your work to it as you go (e.g. one commit per finished
task).

**How you check yourself.**
Every function has self-checks (properties that must hold for any input), and from
Task 1 on you have a slow but obviously correct method, `brute_force`, to compare with.
Run the self-checks on both model files and on several short sequences of your own, not
only on the examples.

Reading: Jones & Pevzner, *An Introduction to Bioinformatics Algorithms*, Chapter 11
(Hidden Markov Models, pp. 387–408); Compeau & Pevzner, *Bioinformatics Algorithms: An
Active Learning Approach*, 2nd ed., Vol. II, Chapter 10: hidden Markov models, the
Viterbi algorithm, the Forward-Backward algorithm.

### Files

- `weather_model.json`: the weather HMM of the lecture (hidden H = high, L = low air
  pressure; seen s = sunny, r = rainy). All examples below use it.
- `toy_model.json`: the model of the paper handout (hidden M = membrane, S = soluble;
  seen h = hydrophobic, p = polar, c = charged residue).
- `sequences.fasta`: eight sequences over h, p, c, of increasing length, **simulated**
  from `toy_model.json`.
- `true_paths.fasta`: for each of them, the hidden path (M/S) that generated it, with
  the same id.

### Words used below

- A **model file** is JSON with the whole model, as in the lecture's definition
  HMM = (states, alphabet Σ, π, A, E). It has five keys: `"states"` (the list of hidden
  states), `"alphabet"` (the list of letters the states emit), `"pi"` (start
  probabilities: state → probability), `"a"` (transitions: state → state → probability)
  and `"e"` (emissions: state → letter → probability). Read it with `json.load`
  (`import json`); you get two lists and three dicts of dicts.
- The **order of the states** is the order of the `"states"` list. It decides ties
  (see Task 2).
- A sequence `x` and a hidden path `z` are both strings of the same length; `z[i]` is
  the state that emitted `x[i]`. Positions are **0-based**, as in Python.
- **Logarithms** are base 2 (`math.log2`), as in the lecture: scores in bits. A
  probability of 0 has log2 = −∞, in Python `-math.inf`; `math.log2(0)` raises an
  error, so you will write a small helper.

---

## Task 1: The model in code

First do the paper part of Task 1 on the handout.

The model is read once and then used for every sequence: that is why it is a class
(just as `FMIndex` in Exercise 06 was built once for every read). `log2`, `read_model`,
`log_sum` and `agreement` are plain functions; everything else is a method of `HMM`
(e.g. `m.viterbi(x)`).

#### `log2(p)`
- **Input:** `p`, a probability (a float ≥ 0).
- **Output:** `float`, `math.log2(p)`, or `-math.inf` when `p` is 0.

#### `HMM(states, alphabet, pi, a, e)`
- **Input:** the two lists and the three dicts of a model file.
- **Stores:** `self.states`, `self.alphabet`, `self.pi`, `self.a` and `self.e`, and the
  same three tables in log2: `self.log_pi`, `self.log_a` and `self.log_e`, nested in the
  same way (`self.log_a["H"]["L"]` = log2 a(H→L)). Compute them once, here, with your
  `log2`; every algorithm from `log_joint` on uses only them, so no other method calls
  `log2`.
- **Hint:** check in the constructor that the tables fit the lists: `pi`, `a`, `e` and
  every row of `a` have exactly the keys of `states`, every row of `e` exactly the
  letters of `alphabet` (compare sets), and `pi` and every row of `a` and `e` sum to 1
  (`math.isclose`, not `==`: 0.1 + 0.2 is not exactly 0.3 in floating point).

#### `read_model(path)`
- **Input:** `path` of a model file.
- **Output:** an `HMM`.
- **Example:** `read_model("weather_model.json").states` → `["H", "L"]`,
  `.alphabet` → `["s", "r"]`.

#### `joint_probability(x, z)`
- **Input:** a sequence `x` and a path `z` of the same length.
- **Output:** `float`, P(x, z) = π(z₀)·e(z₀, x₀) · a(z₀→z₁)·e(z₁, x₁) · … ·
  a(zₙ₋₂→zₙ₋₁)·e(zₙ₋₁, xₙ₋₁).
- **Example** (weather): `joint_probability("sr", "HL")` → 0.054 (0.5 · 0.9 · 0.2 · 0.6).

#### `log_joint(x, z)`
- **Input:** as `joint_probability`.
- **Output:** `float`, log2 P(x, z), computed as a **sum** of the log2 of every factor,
  taken from `self.log_pi`, `self.log_a` and `self.log_e` (not as the log2 of the
  product).
- **Example** (weather): `log_joint("sr", "HL")` → about −4.21.
- **Self-check:** `log_joint(x, z)` equals `log2(joint_probability(x, z))` on both model
  files and several sequences and paths (`math.isclose`).

#### `brute_force(x)`
- **Input:** a sequence `x` (short!).
- **Output:** `tuple[str, float, float]`: the most likely path, its probability, and the
  **sum** of `joint_probability(x, z)` over all paths z.
- **Hint:** `itertools.product(self.states, repeat=len(x))` gives every path as a tuple
  of states; `"".join(...)` turns it into a string.
- **Example** (weather): `brute_force("ssrssrrrss")` → `("HHHHHLLLHH", 5.53e-05, 0.000822)`
  (rounded; the lecture's numbers).

---

## Task 2: Viterbi

First do the paper part of Task 2 on the handout.

#### `viterbi(x)`
- **Input:** a sequence `x`.
- **Output:** `tuple[str, float]`, the most likely hidden path and its log2 probability.
- **Hint:** work in log2 from the start. Keep two lists of dicts, one entry per
  position: `v[i][k]` = log2 probability of the best path that ends in state k at
  position i, and `back[i][k]` = the state before k on that path. Fill column 0 from
  `log_pi` and `log_e`, then each column from the previous one (lecture: Viterbi
  recurrence, now with + instead of ·), then take the best state in the last column and
  follow `back` to position 0. Collect the path backwards and reverse it.
- **Ties:** compare with `>` and go through the states in the order of
  `self.states`, so that the first state wins a tie (in every max and in the last
  column). Then everyone gets the same path.
- **Example** (weather): `viterbi("ssrssrrrss")` → `("HHHHHLLLHH", -14.14...)`.
- **Self-checks:**
  1. The path and its probability equal `brute_force` (`2 ** score` vs. the probability,
     `math.isclose`) on both model files and many short sequences (e.g. every sequence
     of length 5 over the alphabet: `itertools.product` again).
  2. The score equals `log_joint(x, path)`.
  3. For the handout sequence and `toy_model.json`, print `v[i][k]` rounded to 2
     decimals for every cell: they are the numbers of your paper table (your paper table
     adds rounded values, so the last columns may differ by a few hundredths).

*Think about it:* make a copy of `viterbi` that multiplies plain probabilities instead
of adding logs, and run both on every sequence of `sequences.fasta` (read them with
Biopython: `from Bio import SeqIO`, `SeqIO.parse("sequences.fasta", "fasta")`,
`str(record.seq)`). From which length does the plain version break? What does it return
then, and why is the path it returns not reliable? Roughly how many bits per position
does the best path cost here (the log2 score divided by the length)?

---

## Task 3: Forward, Backward, posterior

First do the paper part of Task 3 on the handout.

#### `log_sum(a, b)`
- **Input:** `a`, `b`, two log2 values (possibly `-math.inf`).
- **Output:** `float`, log2(2^a + 2^b), computed without leaving log space:
  max + log2(1 + 2^(min − max)).
- **Hint:** handle −∞ first (if one of them is −∞, return the other one), otherwise
  `-inf - (-inf)` gives `nan`.
- **Examples:** `log_sum(3, 3)` → 4.0; `log_sum(0, -math.inf)` → 0.0;
  `log_sum(-math.inf, -math.inf)` → `-inf`.

#### `forward(x)`
- **Input:** a sequence `x`.
- **Output:** `tuple[list[dict], float]`: the table `f`, `f[i][k]` = log2 P(x₀ … xᵢ,
  zᵢ = k), and log2 P(x).
- **Hint:** the same loop as `viterbi`, with the max replaced by a sum (`log_sum`) and no
  `back`. `log_sum` adds two values: start each sum at `-math.inf` and add the
  candidates one by one with `log_sum`. log2 P(x) = `log_sum` over the last column.
- **Example** (weather): the log2 P(x) of `"ssrssrrrss"` is about −10.25 (P(x) = 0.000822).

#### `backward(x)`
- **Input:** a sequence `x`.
- **Output:** `tuple[list[dict], float]`: the table `b`, `b[i][k]` = log2 P(xᵢ₊₁ … xₙ₋₁
  | zᵢ = k), and log2 P(x) computed from it.
- **Hint:** fill from the end: the last column is 0 (= log2 1) for every state; then
  `b[i][k]` = `log_sum` over l of `log_a[k][l]` + `log_e[l][x[i + 1]]` + `b[i + 1][l]`.
  At the end, log2 P(x) = `log_sum` over k of `log_pi[k]` + `log_e[k][x[0]]` + `b[0][k]`.

#### `posterior(x)`
- **Input:** a sequence `x`.
- **Output:** `list[dict]`, for every position i and state k the probability
  P(zᵢ = k | x) = 2^(f[i][k] + b[i][k] − log2 P(x)) (a plain probability, not a log).
- **Example** (weather): `posterior("ssrssrrrss")[2]["L"]` → about 0.65.

#### `posterior_path(x)`
- **Input:** a sequence `x`.
- **Output:** `str`, at every position the state with the highest posterior
  probability (ties: the first state, as in Task 2).
- **Example** (weather): `posterior_path("ssrssrrrss")` → `"HHLHHLLLHH"` (the lone rainy
  day of the lecture).
- **Self-checks** (Task 3 together, on both model files and many short sequences):
  1. `forward` and `backward` give the same log2 P(x).
  2. 2 to the power of it equals the sum returned by `brute_force`.
  3. Every column of `posterior` sums to 1.
  4. *(optional)* P(zᵢ = k | x) equals the sum of `joint_probability` over the paths with
     k at position i, divided by the sum over all paths (brute force).
  5. For the handout sequence, `2 ** f[i][k]` and `2 ** b[i][k]` equal your paper
     Forward and Backward values, and the posterior your paper posterior (up to
     rounding).

### Decode the simulated sequences

1. Read `sequences.fasta` and `true_paths.fasta` with `SeqIO.parse`; match them by
   `record.id`.
2. Decode every sequence with `viterbi` and with `posterior_path`.
3. Write `agreement(z1, z2)`: the fraction of positions where two paths of the same
   length have the same state. Print, for every sequence, its id, its length, and the
   agreement of each decoding with the true path.
4. **Self-checks:** a path agrees with itself 100 %; the Viterbi score equals
   `log_joint` of the Viterbi path on every sequence; `forward` and `backward` agree on
   every sequence.

*Think about it:* (a) compute `joint_probability` of the posterior path of the handout
sequence and compare it with the probability of the Viterbi path. What does each path
optimise? (b) On `sequences.fasta`, which of the two decodings agrees more often with
the true path, position by position? Why is that exactly what you should expect? When
would you still prefer the Viterbi path?
