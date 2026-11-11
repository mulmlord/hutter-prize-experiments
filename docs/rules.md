# Rules and budgets

Sources: <http://prize.hutter1.net/> and <http://prize.hutter1.net/hrules.htm> (checked 2026-09-27).

## Task
- The compressor `comp9` (size S1) reads enwik9 (10^9 bytes) and writes a self-extracting `archive9` (size S2).
- Running `archive9` with no input from any other source must reproduce enwik9 byte for byte.
- **Score `S = S1 + S2`.** Anything either program needs (dictionaries, weights) counts, so data
  both programs carry is paid for twice.
- Prize: EUR 500,000 × (1 − S/L), where L is the previous record. The minimum claim is 1%,
  so `S ≤ floor(0.99 · L)`.

## Resources (each program, separately)
- One CPU core, no GPU.
- Less than 10 GB RAM and less than 100 GB of temporary disk.
  The rules don't say whether GB means 10^9 or 2^30 bytes, and it matters: zmix's reported compression
  peak RSS of 9,782,592 kB (GNU time reports KiB) is 10.017 × 10^9 bytes, over the decimal limit and under the binary one (10.74 × 10^9).
  `config/targets.json` uses 10^9, so `score.py table` flags anything above it. Our own entry should stay under 10^10 bytes.
- Time: less than 70,000/T hours, where T is the test machine's Geekbench 5 single-core score.
  We track `W = hours × T ≤ 70,000`.

| Machine | GB5 single | Hour limit |
|---|---|---|
| AMD Ryzen 9 5900X (Zen 3) | 1648 (zmix measured 1724) | 42.5 h (40.6 h at 1724) |
| Intel Core i7-1165G7 | ~1427 | 49.1 h |
| ours: Ryzen 7840HX (Zen 4) | 1753 (median of 1809, 1705, 1753) | 39.9 h |

zmix 1.0 compresses at W = 62,905, about 90% of the budget. By its own README it does
**not** meet the limit on the Intel machine. Speed is the binding constraint.

### Our laptop (T003, 2026-09-27)
- The CPU sits at its thermal limit: Tctl reads 96–101 °C even near idle with the fans at ~7,400 RPM.
  A sustained single-core load holds about **4.55 GHz, not the 5.18 GHz peak**, and a second
  load on the other CCD (cores 2 + 6) doesn't lower it measurably over 2 minutes.
- GB5 runs [24670020](https://browser.geekbench.com/v5/cpu/24670020), [24670024](https://browser.geekbench.com/v5/cpu/24670024)
  and [24670027](https://browser.geekbench.com/v5/cpu/24670027) (5.5.1 Tryout, unpinned, back to back). The first run,
  on a cooler CPU, scored highest, so heat costs a few percent even in a 2-minute benchmark.
- Our T = 1753 is close to the 5900X (1648 official, 1724 as zmix measured), so local hours map almost 1:1
  to judge hours: a local run needs to finish in under 39.9 h.
- Geekbench runs in short bursts and may see higher clocks than a 36-hour run does. If so, our T is
  too high and our local W overstates the judges' W, which is the safe direction. Compare the
  `sensors.csv` median of a long run with the clock seen during Geekbench.

## Publication
- Source code under an OSI licence before payout. Upstream code is GPL-3.0, so our code will be too.
- 30-day public comment period before the award.

## Portability rules we impose on ourselves
- No `RCPPS`/`RSQRTPS` in the binary. Build with `-mrecip=none`; `scripts/neif_gate.sh` checks for them.
  Their results differ between Intel and AMD, and an archive made on one vendor would then decode to garbage on the other.
- **No AVX-512 in shipped code paths.** The Zen 3 judge machine lacks it, even though our Zen 4 laptop has it.
  Local timings with AVX-512 enabled are meaningless.
- Floating point must be bit-reproducible: no `-ffast-math`, a fixed summation order, and no
  threads in the coding path.

## Byte arithmetic worth remembering
- 1% of zmix's S is about 993 KB.
- Model data costs 2 bytes of S per byte shipped. A 4-bit model costs about 1 byte of S per parameter.
- Code costs twice too: zmix's packed program image is 160,884 bytes; weights are 2,815,630 (see docs/landscape.md).
