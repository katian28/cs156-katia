# Session 12: Linear Regression with Basis Functions

## Overview

Linear regression is more powerful than it looks: by transforming the basis a feature is represented in (polynomial powers, radial basis functions), a model that's linear in its *parameters* can fit curves that are highly nonlinear in the original variable. This session covers the linear-in-parameters trick, basis function shapes, ridge/lasso regularization, and applies all of it to a real, messy dataset: 5-minute temperature readings from Yosemite Village, CA (NOAA U.S. Climate Reference Network, 2011-2016).

## Question 1: Basic Questions

**Linearity in Parameters.** A function is linear in its *parameters* if no parameter (the $\beta$'s) is squared or multiplied by another parameter. It's linear in its *variables* if no variable ($x$) is squared or raised to a power. These are independent:

| Function | Linear in parameters? | Linear in variables? |
|---|---|---|
| $y = \beta_0 + \beta_1 x + \beta_2 x$ | Yes | Yes |
| $y = \beta_0 + \beta_1 x + \beta_2 x^2$ | Yes | No |
| $y = \beta_0 + \beta_1 x + \beta_2^2 x$ | No | Yes |
| $y = \beta_0 + \beta_1 x + \beta_2^2 x^2$ | No | No |

Row 2 (linear in parameters, not variables) is the trick basis function regression relies on: you can transform $x$ into $x^2$, $x^3$, or a distance from some center, and still solve the fit with ordinary linear regression, as long as the coefficients themselves aren't touched.

**Basis Functions.**
- Radial basis function $f(x) = w_0 + w_1|x-c|$: a V-shape centered at $c$. $c$ sets the V's horizontal position, $w_0$ shifts it vertically, $w_1$ sets the steepness (and flips it upside-down if negative). It transforms $x$ into a distance-from-a-reference-point, and a sum of several RBFs at different centers can approximate complex, locally-varying curves.
- Polynomial basis function $f(x) = w_0 + w_1x + w_2x^2 + w_3x^3 + w_4x^4$: expands a single $x$ into a 5-dimensional feature vector $[1, x, x^2, x^3, x^4]$. The curve can bend up to 3 times, but the fit itself is still ordinary linear regression in that expanded space.

**Regularization.** Ridge adds an L2 penalty (sum of squared coefficients), shrinking all coefficients toward zero without usually zeroing any out, useful when many basis functions overlap and would otherwise blow up fitting noise. Lasso adds an L1 penalty (sum of absolute coefficients), which can zero coefficients out entirely, automatic feature/basis-function selection.

## Question 2: Core Questions — Yosemite temperature data

### Data columns

The starter code reads `usecols=[1, 2, 8]`. Matched against the USCRN README's field list:

- **Column 1 → `UTC_DATE`** (YYYYMMDD)
- **Column 2 → `UTC_TIME`** (HHmm, end of the 5-minute observation window)
- **Column 8 → `AIR_TEMPERATURE`** (°C, averaged over that window)

The file is fixed-width, 23 columns total (station ID, UTC and local date/time, lat/lon, temperature, precipitation, solar radiation, surface temp, humidity, soil readings, wind). Missing values use sentinels like `-9999`, which is why the starter code filters with `valid = data[:, 2] > -1000`.

### Why the linear model performs so poorly

The straight-line fit on time-of-day alone gets **R² ≈ 0.017**. Two effects get pooled into one x-axis and both defeat a line:

1. **The daily cycle is a hump, not a slope.** Temperature rises through the day and falls overnight, no single slope tracks both halves.
2. **Six years of seasons get pooled into one x-value.** Minutes-since-midnight strips out the date, so 2 PM in January and 2 PM in July land on the same x-coordinate despite a 20+ degree difference. I checked this directly: time-of-day-only signal has standard deviation ≈ 1.65°C against a full-dataset standard deviation of 7.82°C, so time-of-day alone caps out near R² ≈ 0.04 no matter the basis function shape.

Fixing this needs a basis function that can express a hump (RBFs can), and it needs the date itself as a second input dimension, not just time-of-day, so the model can separate "what time of day" from "what season/year."

### Model and metric

I used **R²**, matching both the starter code's own `.score()` calls and the class benchmark stated in the Extension Questions. Final model: a 2D RBF over `(minute-of-day, day-index-since-2011-01-01)`, ~4700 centers on a grid (15 along the daily cycle, every 7 days across the 6-year span), `sigma_min=80` minutes, `sigma_day=6` days, `Ridge(alpha=0.1)`.

| | R² |
|---|---|
| Straight line (time-of-day only) | 0.017 |
| 2D RBF + Ridge, train | 0.839 |
| 2D RBF + Ridge, held-out test | **0.829** |

## Question 4: Extension Questions

**What performance were you able to achieve?**

Test R² ≈ 0.83 (train R² ≈ 0.84), beating the class benchmark of 0.82. The hyperparameter search (notebook, Extension section) swept center count, grid spacing, RBF width, and ridge alpha; the winning config used ~4700 centers, `sigma_min=80`, `sigma_day=6`, `alpha=0.1`. Fewer/coarser centers (1500-2400) topped out around R² ≈ 0.73, confirming the assignment's hint that too few RBFs was the likely failure mode, the real fix was adding date as a second input axis, not just more centers on time-of-day alone.

**What applying lasso to this model might do**

With ~4700 densely-placed, overlapping centers (neighbors 7 days apart with a 6-day width overlap heavily), many are redundant. Lasso's L1 penalty would zero out most of those redundant coefficients, keeping only centers that each explain genuinely distinct regions of the (time, date) grid. That would shrink the model to a sparse subset, trading a little R² for a smaller, faster, more interpretable model, and it would show directly which regions of the grid actually carry information versus which were just padding.

## Files

```
PCW/Session 12 - Basis Functions/
├── README.md
├── pcw_lesson_12.ipynb
└── data/
    ├── CRNS0101-05-2011-CA_Yosemite_Village_12_W.txt
    ├── CRNS0101-05-2012-CA_Yosemite_Village_12_W.txt
    ├── CRNS0101-05-2013-CA_Yosemite_Village_12_W.txt
    ├── CRNS0101-05-2014-CA_Yosemite_Village_12_W.txt
    ├── CRNS0101-05-2015-CA_Yosemite_Village_12_W.txt
    └── CRNS0101-05-2016-CA_Yosemite_Village_12_W.txt
```

Data source: [NOAA U.S. Climate Reference Network, subhourly01 product](https://www1.ncdc.noaa.gov/pub/data/uscrn/products/subhourly01).

## How to Run

```bash
cd "PCW/Session 12 - Basis Functions"
jupyter nbconvert --to notebook --execute --inplace pcw_lesson_12.ipynb
```
