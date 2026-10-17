# NSE-Calibrated PINN Option Pricing & Hedging Engine
## Master Architecture, Data Specification, and Development Roadmap

**Document purpose:** Single source of truth for the project’s research design, data conventions, software architecture, phase dependencies, and Codex implementation workflow.

**Project scope:** NIFTY European call (CE) and put (PE) option pricing using a Physics-Informed Neural Network (PINN), with classical pricing benchmarks, market validation, Greeks, and daily historical delta hedging.

---

## 1. Project Objective

Build an NSE-calibrated quantitative engine for NIFTY index options. The system will combine NIFTY historical prices, CE/PE option-chain observations, and RBI Treasury-bill yield observations to construct market snapshots, derive pricing parameters, solve the Black–Scholes pricing PDE using a PINN, compute Greeks, compare model prices with classical benchmarks and observed market prices, and evaluate daily delta-hedging experiments.

The intended research pipeline is:

```text
Raw market data
    ↓
Canonical data layer
    ↓
Contract universe and market snapshots
    ↓
Market parameter calibration
    ↓
Black–Scholes / Monte Carlo benchmarks
    ↓
Baseline PINN
    ↓
Parametric NSE-calibrated PINN
    ↓
Validation and Greeks
    ↓
Daily historical delta hedging
    ↓
API, frontend, and deployment
```

The quantitative engine must remain independent of the raw CSV layout. The UI and deployment work come only after the numerical engine has passed its validation gates.

---

## 2. Actual Data Inventory and What It Supports

The four supplied source files are the source of truth for data structure. The original files must remain immutable; all normalization and derived fields belong in processed outputs.

| Dataset | Reported structure | Intended role |
|---|---:|---|
| `nifty50_historical.csv` | 4,645 daily rows, 7 fields; Jan 2008–Oct 5, 2026 | Spot history, returns, realized volatility |
| `nifty_ce_options_model_ready.csv` | 995,104 contract-day rows, 17 fields; Jan 2020–Oct 5, 2026 | Call prices, strikes, expiries, open interest and liquidity diagnostics |
| `nifty_pe_options_model_ready.csv` | 1,062,839 contract-day rows, 17 fields; Jan 2020–Oct 5, 2026 | Put prices, strikes, expiries, open interest and liquidity diagnostics |
| `rbi_tbill_yield_15_91d_monthly.csv` | 78 monthly observations; Jan 2020–Jun 2026 | Potential risk-free-rate input, subject to source-unit verification |

The option datasets together contain approximately 2.06 million contract-day rows. The CE and PE source files share the same conceptual schema.

### 2.1 Option fields

- **Contract identity:** Symbol, Date, Expiry, Option type, Strike Price
- **Price fields:** Open, High, Low, Close, LTP, Settle Price
- **Market activity:** No. of contracts, Turnover, Premium Turnover, Open Int, Change in OI
- **Underlying:** Underlying Value

### 2.2 Data realities that constrain the design

1. **Daily end-of-day data only.** The option files have no intraday timestamp. Version 1 must not claim intraday pricing or intraday hedging.
2. **`Close` is the primary observed option price.** It is the only universally populated option price field in the reported profiling. Other price and activity fields are diagnostics, not required model inputs.
3. **CE and PE should be unified.** Normalize both files into one `option_eod` table with a canonical `option_type` field (`CE` or `PE`). Keep the source files unchanged.
4. **The date ranges differ.** NIFTY and option data extend to October 5, 2026, but the supplied RBI series ends in June 2026. Do not invent a contemporaneous rate for July–October and label the result fully RBI-calibrated.
5. **NIFTY history predates option data.** Use 2008–2019 for underlying-return history, volatility warm-up, and long-horizon statistics; do not claim option-market validation for years without option records.
6. **Expiry is a date, not a categorical substitute for time.** Compute time to maturity explicitly and classify contracts by whether they are expired, at maturity, or active.
7. **RBI scale is policy-controlled.** The export reports the unit as “Indian rupee,” while official RBI rate publications present Treasury-bill yields as percentages. The approved policy converts the source values by `0.01`, preserves the original metadata, and records the prior-month as-of rule.

### 2.3 Reported price-field missingness

| Field | CE missing | PE missing | Design implication |
|---|---:|---:|---|
| `Close` | 0% | 0% | Primary market-price benchmark |
| `Underlying Value` | 0% | 0% | Consistency diagnostic against NIFTY close |
| `LTP` | 47.24% | 42.88% | Optional diagnostic only |
| `Settle Price` | 0.14% | 1.03% | Optional diagnostic only |
| `Open` / `High` / `Low` | 57.89% | 56.40% | Do not make them required |

These percentages are the reported profiling results and should be rechecked by the ingestion tests rather than treated as permanent guarantees.

---

## 3. Canonical Data and Market Observation Contracts

### 3.1 Canonical datasets

```text
underlying_daily
option_eod
risk_free_monthly
market_snapshot
```

The first three are normalized source datasets. `market_snapshot` is a derived, reusable pricing-scenario table.

### 3.2 Canonical option table

The unified `option_eod` table should retain, where present:

```text
date
expiry
symbol
option_type             # CE / PE
strike
open
high
low
close
ltp
settle_price
contracts
turnover
premium_turnover
open_interest
change_in_oi
underlying_value
```

Normalize source column names and types consistently, but do not replace a missing source value with another field (for example, never silently substitute `LTP` or `Settle Price` for `Close`).

### 3.3 Market observation unit

One option observation is identified by:

```text
valuation_date × expiry × strike × option_type
```

A market snapshot should contain the information required to price and assess a contract, including:

- valuation date and expiry;
- option type and strike \(K\);
- spot \(S\);
- time to maturity \(T\);
- observed option `Close` for validation;
- calibrated risk-free rate \(r\), when valid;
- selected realized volatility \(\sigma\), when available;
- optional carry \(q\), only when supported by the chosen model;
- moneyness \(S/K\) and/or \(\log(S/K)\);
- data-quality and liquidity flags.

### 3.4 Time-to-maturity and contract state

Use a documented day-count basis:

\[
T = \frac{\text{Expiry Date} - \text{Valuation Date}}{\text{day-count basis}}
\]

The day-count convention must be configurable or explicitly frozen in the project contract.

- **Expired:** \(T < 0\). Exclude from the active pricing domain.
- **Maturity day:** \(T = 0\). Retain for terminal-payoff tests.
- **Active:** \(T > 0\). Eligible for the interior pricing domain.

Do not silently discard records in the raw or canonical ingestion layer merely because they are expired or at maturity. Apply contract eligibility in the snapshot/pricing layer and record the reason.

---

## 4. Calibration and Leakage-Prevention Design

The project must separate the information used to define model parameters from the market prices used to evaluate the model.

### 4.1 Parameter sources

| Parameter | Primary source | Rule |
|---|---|---|
| Spot \(S\) / \(S_0\) | NIFTY historical `Close` | Canonical underlying series |
| Strike \(K\) | Option contract metadata | Numeric strike |
| Maturity \(T\) | Valuation date and expiry | Explicit day-count convention |
| Risk-free rate \(r\) | RBI 15–91-day T-bill series | Resolve units and align as-of without look-ahead |
| Realized volatility \(\sigma\) | Historical NIFTY log returns | Rolling, annualized, backward-looking windows |
| Optional carry \(q\) | Later CE–PE parity diagnostics or other justified source | Extension, not a mandatory initial input |
| Market price | Option `Close` | Validation target; not part of the primary pure-PDE PINN loss |

Compute NIFTY log returns as:

\[
u_t = \ln\left(\frac{S_t}{S_{t-1}}\right)
\]

Initially investigate 20-, 60-, and 252-trading-day rolling annualized volatility windows. The selected calibration policy should be justified by tenor and validation experiments, not chosen after looking at held-out market errors.

### 4.2 Risk-free-rate policy

- Parse the RBI pipe-delimited file according to its actual format.
- Preserve the original value and unit metadata.
- Resolve the reported unit/scale ambiguity before treating the series as a valid numerical rate.
- Use a configurable, leakage-safe as-of alignment rule.
- Do not forward-fill into a date before an observation was available.
- Do not fabricate a July–October 2026 rate. Unless an approved additional source is added, fully calibrated historical market validation should end at the latest supported RBI observation (reported as June 2026).

### 4.3 Keep market validation separate from primary PINN training

The primary PINN learns the PDE and boundary/terminal conditions. Observed option `Close` is reserved for evaluation and market diagnostics. If a future experiment deliberately uses market prices in a calibration or supervised loss, label it as a separate experiment and keep it distinct from the pure-PDE baseline.

This distinction is essential to avoid presenting a model that learns market labels as if it had independently solved the pricing PDE.

---

## 5. Quantitative Architecture

```text
┌──────────────────────── RAW DATA SOURCES ────────────────────────┐
│ NIFTY daily history │ CE options │ PE options │ RBI T-bill data  │
└──────────────────────────────┬───────────────────────────────────┘
                               ↓
┌──────────────────── INGESTION AND VALIDATION ────────────────────┐
│ Schema/date/type checks, missingness, duplicates, ranges, units  │
└──────────────────────────────┬───────────────────────────────────┘
                               ↓
┌──────────────────── CANONICAL DATA LAYER ────────────────────────┐
│ underlying_daily │ option_eod │ risk_free_monthly                │
└──────────────────────────────┬───────────────────────────────────┘
                               ↓
┌──────────────────── MARKET SNAPSHOT BUILDER ─────────────────────┐
│ Date, S, K, expiry, T, CE/PE, Close, quality and liquidity flags  │
└──────────────────────────────┬───────────────────────────────────┘
                               ↓
┌────────────────────── CALIBRATION LAYER ─────────────────────────┐
│ NIFTY → S, realized volatility │ RBI → r │ CE/PE → diagnostics    │
└──────────────────────────────┬───────────────────────────────────┘
                               ↓
┌──────────────────────── QUANTITATIVE CORE ───────────────────────┐
│ Black–Scholes │ Monte Carlo │ optional finite-difference PDE      │
│ Baseline PINN │ NSE-calibrated parametric PINN                    │
└──────────────────────────────┬───────────────────────────────────┘
                               ↓
┌──────────────────────── VALIDATION ENGINE ───────────────────────┐
│ PINN vs BS/MC │ PINN vs NSE Close │ PDE residual │ convergence    │
│ Error slices by CE/PE, moneyness, maturity, date, liquidity       │
└──────────────────────────────┬───────────────────────────────────┘
                               ↓
┌──────────────────── GREEKS AND HEDGING ───────────────────────────┐
│ Greeks via derivatives │ daily historical delta-hedging P&L       │
└──────────────────────────────┬───────────────────────────────────┘
                               ↓
┌──────────────────────── APPLICATION LAYER ───────────────────────┐
│ FastAPI inference endpoints │ interactive frontend and plots     │
└──────────────────────────────────────────────────────────────────┘
```

Calibration and market validation are separate logical branches: NIFTY/RBI/contract metadata produce model parameters; observed option `Close` is retained as a market benchmark.

---

## 6. PINN Research Design

### 6.1 Baseline: fixed-parameter PINN

Start with a controlled model:

\[
V_\theta(S,t)
\]

The network is a fully connected neural network. Automatic differentiation provides:

\[
V_t,\qquad V_S,\qquad V_{SS}
\]

For the no-dividend Black–Scholes baseline, the PDE residual is:

\[
R =
V_t +
\frac{1}{2}\sigma^2 S^2 V_{SS}
+ r S V_S - rV
\]

If a carry/dividend extension is enabled, the PDE and its parameters must be changed consistently and tested separately.

The loss combines:

```text
PDE residual loss
+ terminal/payoff loss
+ boundary-condition loss
```

The exact terminal condition depends on option type:

\[
V(S,T)=\max(S-K,0) \quad \text{for a call}
\]

\[
V(S,T)=\max(K-S,0) \quad \text{for a put}
\]

The baseline must be tested against the analytical Black–Scholes solution before market observations are introduced into any training objective.

### 6.2 Parametric / universal PINN

The fixed-parameter PINN is a controlled research baseline. The intended final model should generalize across many strikes, maturities, and calibrated market scenarios, rather than requiring a separately trained network for every contract.

A conceptual parameterized input is:

\[
V_\theta\left(\frac{S}{K},\tau,\sigma,r,q,\text{option type}\right)
\]

with a normalized output such as:

\[
\frac{V}{K}
\]

Here, \(\tau\) is the appropriately normalized time coordinate. The exact input encoding, normalization, architecture, and carry handling should be documented and validated before freezing the model.

### 6.3 Experiment modes

- **Mode A — Pure PDE:** PINN learns from the PDE and boundary/terminal conditions only.
- **Mode B — Market-parameterized inference:** NIFTY supplies spot and realized-volatility inputs, RBI supplies the validated rate, and option metadata supplies strike and maturity.
- **Mode C — Market diagnostics:** observed CE/PE prices are used for implied-volatility and parity/carry diagnostics, not silently mixed into the pure-PDE training loss.

---

## 7. Repository Layout

```text
nse-pinn-option-engine/
├── data/
│   ├── raw/
│   │   ├── nifty50/
│   │   ├── options/
│   │   └── rbi/
│   └── processed/
│       ├── underlying_daily/
│       ├── options_eod/
│       ├── risk_free/
│       └── market_snapshots/
├── configs/
│   ├── data.yaml
│   ├── calibration.yaml
│   ├── pinn.yaml
│   └── experiments.yaml
├── src/
│   ├── data/
│   ├── calibration/
│   ├── contracts/
│   ├── quantitative/
│   │   ├── black_scholes/
│   │   ├── monte_carlo/
│   │   └── numerical_pde/
│   ├── pinn/
│   ├── validation/
│   ├── greeks/
│   ├── hedging/
│   ├── experiments/
│   └── api/
├── tests/
│   ├── data/
│   ├── calibration/
│   ├── quantitative/
│   ├── pinn/
│   └── validation/
├── notebooks/
│   └── exploration/
├── experiments/
│   ├── baseline/
│   ├── market_calibration/
│   ├── pinn/
│   └── validation/
├── frontend/
├── models/
├── reports/
└── README.md
```

CSV is the raw input format, not the internal runtime format. Canonicalize large option files once into a columnar, query-friendly format such as Parquet, while keeping the raw CSVs immutable.

---

## 8. Development Roadmap and Phase Gates

# PHASE 0 — Project Contract & Architecture Freeze

### Objective

Freeze the mathematical, data and software contracts before writing the main implementation.

### Define

- Exact project objective
- Supported option types: CE and PE
- Daily EOD valuation convention
- Spot definition
- Time-to-maturity convention
- Risk-free-rate alignment rule
- Volatility-calibration methodology
- Market-price validation methodology
- Leakage-prevention policy
- PINN input/output contract
- Benchmark interfaces
- Experiment configuration system

### Critical rules

1. Raw CSV files remain immutable.
2. Option `Close` is the primary observed market-price field.
3. The observed option price is not used in the primary PDE-training loss.
4. RBI rate alignment must be leakage-safe.
5. Option contracts must satisfy an explicit active-contract rule.
6. July–October 2026 cannot be treated as fully RBI-calibrated historical validation unless the rate source is extended.
7. The PINN must operate independently of CSV structure.

### Output

A frozen technical specification and repository skeleton.

### Exit criterion

Every downstream module has a clearly defined input and output.

---

# PHASE 1 — Data Ingestion & Canonical Data Layer

### Objective

Convert the four supplied datasets into validated canonical datasets.

### Inputs

#### NIFTY

- Date
- Open
- High
- Low
- Close
- Shares Traded
- Turnover

#### CE / PE

- Symbol
- Date
- Expiry
- Option type
- Strike Price
- Open
- High
- Low
- Close
- LTP
- Settle Price
- No. of contracts
- Turnover
- Premium Turnover
- Open Int
- Change in OI
- Underlying Value

#### RBI

- Element
- Term to maturity
- Time
- Unit
- Value

### Work

- Parse dates
- Normalize column names
- Convert numeric fields
- Handle the RBI pipe-delimited format
- Validate schemas
- Profile missingness
- Detect duplicates
- Validate ranges
- Validate OHLC relationships
- Preserve raw files
- Convert canonical data to efficient storage

### Canonical outputs

```text
underlying_daily
option_eod
risk_free_monthly
```

### Exit criterion

All downstream code works from canonical tables instead of raw CSV formats.

---

# PHASE 2 — Contract Universe & Market Snapshot Builder

### Objective

Create the actual observation units used by the quantitative engine.

Each observation should represent:

```text
valuation_date
    +
option_type
    +
strike
    +
expiry
```

### Derive

\[
S_0
\]

\[
K
\]

\[
T
\]

\[
\text{option type}
\]

\[
\text{observed market price}
\]

### Contract rules

- Exclude expired contracts from the pricing domain.
- Keep maturity-day contracts for terminal-payoff testing.
- Keep only positive time-to-maturity for interior PDE samples.
- Flag extremely illiquid observations.
- Preserve both CE and PE.
- Create moneyness measures:

\[
M=\frac{S}{K}
\]

and/or

\[
\log\left(\frac{S}{K}\right).
\]

### Data-quality flags

Examples:

```text
has_ltp
has_settlement
has_volume
has_oi
has_change_oi
liquidity_quality
price_quality
active_contract
```

### Output

A reusable:

```text
market_snapshot
```

dataset.

### Exit criterion

One row of `market_snapshot` completely describes a pricing scenario.

---

# PHASE 3 — Market Calibration Engine

### Objective

Translate market observations into the parameters required by the pricing model.

### 3.1 Spot

Use the NIFTY historical close as the canonical underlying series.

Retain `Underlying Value` from the option files for consistency diagnostics.

### 3.2 Risk-free rate

Convert the RBI monthly 15–91-day T-bill yield into a daily as-of rate.

The mapping must be configurable and leakage-safe.

### 3.3 Realized volatility

Calculate NIFTY log returns:

\[
r_t=\ln\left(\frac{S_t}{S_{t-1}}\right)
\]

and rolling annualized volatility.

Initially calculate several windows:

```text
20-day
60-day
252-day
```

Then determine which calibration is most appropriate for different option tenors.

### 3.4 Optional carry / dividend diagnostic

Because both CE and PE data exist, create a later module for CE–PE parity diagnostics.

This should initially be treated as a diagnostic/extension rather than a mandatory model input.

### 3.5 Implied volatility

Invert the Black–Scholes price to obtain market-implied volatility from observed option prices.

IV should initially be treated as:

```text
market diagnostic
```

rather than being inserted directly into the primary PINN training target.

### Output

A calibrated market parameter object:

```text
S0
K
T
r
sigma_realized
optional q
market_price
option_type
```

### Exit criterion

The quantitative engine can be called without knowing anything about NSE CSV files.

---

# PHASE 4 — Analytical & Numerical Benchmark Engine

### Objective

Build the classical quantitative models before building the PINN.

### Black–Scholes

Implement:

- Call
- Put
- Delta
- Gamma
- Theta
- Vega
- Rho

and the corresponding PDE.

### Monte Carlo

Implement:

```text
GBM simulation
      ↓
terminal S
      ↓
payoff
      ↓
discounted expectation
```

Test:

- path count
- timestep count
- convergence
- Monte Carlo error

### Optional numerical PDE

Finite difference can be added as an independent numerical benchmark once the first two models are stable.

### Output

A trusted reference engine:

```text
BS price
MC price
BS Greeks
```

### Exit criterion

Monte Carlo converges toward the analytical Black–Scholes result under the same assumptions.

---

# PHASE 5 — Baseline PINN Solver

### Objective

Prove that the PINN can correctly solve the pricing PDE in a controlled environment.

### Network

Initial model:

\[
V_\theta(S,t)
\]

Use a fully connected neural network.

### Automatic differentiation

Compute:

\[
V_t,\qquad V_S,\qquad V_{SS}.
\]

### PDE residual

\[
R=
V_t+
\frac{1}{2}\sigma^2S^2V_{SS}
+rSV_S-rV
\]

with the configured carry extension where applicable.

### Loss components

```text
PDE residual loss
+
terminal/payoff loss
+
boundary-condition loss
```

### Training

- Generate collocation points
- Train network
- Track each loss component
- Track PDE residual
- Test multiple seeds
- Save model checkpoints

### Required sanity checks

Before market data is introduced:

- terminal payoff correctness
- boundary behavior
- convergence
- known Black–Scholes solution
- price surface shape

### Output

A mathematically verified baseline PINN.

### Exit criterion

PINN agrees with analytical Black–Scholes within predefined numerical tolerances.

---

# PHASE 6 — NSE-Calibrated Parametric PINN

### Objective

Turn the controlled PINN into the actual market-calibrated model.

Instead of training one network for one isolated option, build a parameterized pricing model.

### Conceptual input

```text
normalized spot / strike
time-to-maturity
volatility
risk-free rate
optional carry
option type
```

For example:

\[
V_\theta
\left(
S/K,\tau,\sigma,r,q,\text{type}
\right).
\]

### Important design principle

The NIFTY/RBI data defines the **parameter distribution**.

The NSE option Close defines the **market validation target**.

This separation prevents the model from becoming a conventional supervised regression model disguised as a PINN.

### Experiment modes

#### Mode A — Pure PDE

PINN learns from physics and boundary conditions only.

#### Mode B — Market-calibrated parameters

Use:

```text
NIFTY → S, σ
RBI   → r
Option metadata → K, T
```

#### Mode C — Market-implied diagnostics

Use observed CE/PE prices to calculate IV and carry metrics for comparison.

### Output

A reusable NSE-calibrated PINN inference engine.

### Exit criterion

The same trained architecture can price many NIFTY option scenarios without retraining a separate network for every individual contract.

---

# PHASE 7 — Validation, Greeks & Quantitative Experiments

### Objective

Answer:

> Is the PINN actually useful?

rather than merely:

> Did the neural network train?

### Compare

```text
PINN
  ↕
Black-Scholes
  ↕
Monte Carlo
  ↕
Observed NSE option Close
```

### Metrics

- MAE
- RMSE
- relative error
- maximum absolute error
- PDE residual
- convergence error
- stability across seeds

### Slice the error by

- CE vs PE
- moneyness
- maturity
- volatility regime
- date
- liquidity quality

### Greeks

Calculate:

\[
\Delta=V_S
\]

\[
\Gamma=V_{SS}
\]

and the remaining Greeks using the appropriate derivatives/calculations.

Compare:

```text
PINN Greeks
vs
Black-Scholes Greeks
```

### Visualization

Create:

- price vs spot
- price vs maturity
- PINN surface
- Black–Scholes surface
- error surface
- Greek surfaces
- PDE residual map
- training-loss curves
- convergence curves

### Market-validation principle

The main historical experiment should use a proper time-based split.

No random row-wise split across dates.

### Output

A complete quantitative evidence package showing where the PINN succeeds and where it fails.

### Exit criterion

The project has measurable evidence comparing PINN performance with classical and observed market benchmarks.

---

# PHASE 8 — Delta Hedging Engine

### Objective

Turn the calculated Greeks into an actual quantitative hedging experiment.

### Initial strategy

Daily delta hedging.

For each selected historical contract:

```text
Option
  ↓
PINN Delta
  ↓
Hedge position in NIFTY
  ↓
Daily rebalance
  ↓
Hedging P&L
```

### Compare

```text
PINN delta hedge
vs
Black-Scholes delta hedge
```

### Evaluate

- cumulative hedge P&L
- mean hedge error
- hedge-error volatility
- terminal replication error
- performance by maturity
- performance by volatility regime

### Important constraint

Because the supplied data is daily EOD data, Version 1 should be a **daily-rebalanced historical hedge experiment**, not an intraday execution simulator.

### Output

A genuine option-pricing + hedging component.

---

# PHASE 9 — FastAPI, Frontend, Integration & Deployment

### Objective

Only after the quantitative engine is validated, expose it as a product.

### Backend

FastAPI services such as:

```text
/price
/greeks
/compare
/surface
/market-snapshot
/hedge
```

### Frontend

Use:

```text
Next.js
React
Plotly
```

### User selects

```text
Date
Option Type
Strike
Expiry
Model
```

The system retrieves the appropriate calibrated market snapshot.

### Display

```text
PINN Price
Black-Scholes Price
Monte Carlo Price

Delta
Gamma
Theta
Vega
Rho

Pricing Error
PDE Residual
```

plus:

- price surfaces
- Greek surfaces
- model comparison
- hedging results

### Engineering rule

The API should perform **inference**, not unexpectedly retrain a large PINN when the user clicks a button.

### Output

A usable quantitative web application.

---


## Final Dependency Chain

```text
Phase 0  → Project contract and architecture
Phase 1  → Raw data to canonical data
Phase 2  → Canonical data to contract snapshots
Phase 3  → Snapshots to calibrated S, K, T, r, σ
Phase 4  → Black–Scholes and Monte Carlo benchmarks
Phase 5  → Verified fixed-parameter PINN
Phase 6  → NSE-calibrated parametric PINN
Phase 7  → Validation, Greeks, and quantitative experiments
Phase 8  → Daily historical delta hedging
Phase 9  → FastAPI, frontend, integration, deployment
```

## 9. Non-Negotiable Project Rules

1. **Raw-data immutability:** never edit the supplied source CSVs.
2. **Canonical interfaces:** downstream modules consume canonical datasets and typed parameter objects, not raw CSV-specific columns.
3. **Close is the market benchmark:** do not silently substitute LTP, settlement, or another field for `Close`.
4. **No market-price leakage into the pure-PDE PINN:** observed `Close` is the primary market-validation target.
5. **Leakage-safe time alignment:** rates and rolling volatility must use only information available as of the valuation date.
6. **Resolve RBI semantics:** no unverified scale conversion and no invented rate values.
7. **Respect maturity:** expired, maturity-day, and active contracts have different roles.
8. **No intraday claims:** the supplied option data supports daily EOD analysis only.
9. **Time-aware evaluation:** use chronological or walk-forward validation, never random row-wise splits across dates for the main historical experiment.
10. **Benchmark before complexity:** validate PINN behavior against Black–Scholes and Monte Carlo before extending the model.
11. **No premature frontend:** the quantitative core and tests must pass before application integration.
12. **Inference is not training:** API requests should not unexpectedly retrain a PINN.

## 10. Codex Implementation Workflow

Implement the project **one phase per Codex prompt**. Each prompt should state:

- phase objective and scope;
- exact files/modules allowed to change;
- inputs, outputs, and interface contracts;
- constraints and explicitly out-of-scope work;
- required tests and commands;
- expected report location;
- completion status and exit criteria.

After each phase, Codex should create a report under `docs/phase_reports/`, documenting:

1. work completed;
2. files created or modified;
3. design decisions and assumptions;
4. tests run and their actual results;
5. warnings, unresolved questions, and limitations;
6. exit-criterion assessment;
7. recommended next phase.

Do not let Codex silently skip failed tests, weaken assertions merely to get a green run, modify raw datasets, or begin future phases without authorization. A phase may be marked complete with warnings only when the warnings are explicitly recorded and do not invalidate its exit criteria.

## 11. Final Project Definition

> **An NSE-calibrated NIFTY option-pricing and hedging engine that builds daily market snapshots from NIFTY historical data, CE/PE option-chain data, and RBI T-bill observations; calibrates spot, risk-free rate, and realized volatility under explicit data-quality and leakage controls; solves the Black–Scholes pricing PDE using a Physics-Informed Neural Network; computes Greeks; validates against analytical and Monte Carlo benchmarks and observed NSE option prices; and evaluates daily historical delta hedging through a quantitative application.**

The nine phases above form the intended master roadmap. Any change to the data conventions, model assumptions, validation protocol, or phase dependency should be documented explicitly rather than introduced implicitly during implementation.
