# Agrivoltaic Tracking Optimization System

Batch processing tool for NSRDB TMY weather data to optimize single-axis tracker control in agrivoltaic systems. The system co-optimizes solar power generation while ensuring sufficient light (DLI) reaches crops beneath the panels.

## Features

- **Bifacial PV Irradiance Model**: Front and back surface irradiance with view factor geometry
- **Single-Axis Tracker Optimization**: Dynamic tracking with backtracking to avoid inter-row shading
- **Crop Light Constraints**: DLI (Daily Light Integral) or biomass-based constraints
- **Two Operation Modes**:
  - **Forward Mode**: Input DLI target → Output PRR (Power Realization Ratio)
  - **Reverse Mode**: Input target PRR → Search for corresponding DLI value
- **GIS Integration**: Output CSV files ready for ArcGIS heatmap visualization

## Prerequisites

### 1. Python Environment

Recommended: Use the repository's virtual environment (Python 3.10+)

```powershell
# Windows
.\.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 2. Dependencies

```bash
pip install numpy pandas pvlib pyomo matplotlib
```

### 3. Nonlinear Solver

The script requires **IPOPT** solver. Download from [COIN-OR Releases](https://github.com/coin-or/Ipopt/releases) and either:
- Add the `bin` directory to system `PATH`, or
- Specify the executable path via `--solver-executable`

## Usage

### Forward Mode (DLI → PRR)

Standard mode: Specify DLI constraint, optimize for maximum power generation.

```bash
python AnalyzeData/main.py \
  --inputs "AnalyzeData/data/file1.csv" "AnalyzeData/data/file2.csv" \
  --output-dir AnalyzeData/output \
  --solver ipopt
```

### Reverse Mode (PRR → DLI)

New mode: Specify target PRR%, search for the DLI value that achieves it.

```bash
python AnalyzeData/main.py \
  --inputs "AnalyzeData/data/file1.csv" \
  --target-prr 80 \
  --dli-min 15 \
  --dli-max 50 \
  --tolerance 0.5 \
  --output-dir AnalyzeData/output
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--inputs` | `data/data1.csv` | Input NSRDB TMY CSV file(s) |
| `--output-dir` | `AnalyzeData/output` | Output directory for results |
| `--criteria` | `dli` | Constraint type: `dli` or `biomass` |
| `--solver` | `ipopt` | Pyomo solver name |
| `--solver-executable` | None | Path to solver executable |
| `--target-prr` | None | Target PRR% (enables reverse mode) |
| `--dli-min` | 15.0 | DLI search lower bound [mol/m²/day] |
| `--dli-max` | 50.0 | DLI search upper bound [mol/m²/day] |
| `--tolerance` | 0.5 | PRR convergence tolerance [%] |
| `--max-iter` | 20 | Maximum binary search iterations |

## Output Files

### Forward Mode

| File | Description |
|------|-------------|
| `*_timeseries.csv` | Hourly power generation and DLI values |
| `summary.csv` | Solver status, CF%, PRR% for each input file |
| `points_prr.csv` | GIS point layer with PRR values |

### Reverse Mode

| File | Description |
|------|-------------|
| `*_timeseries.csv` | Hourly data with additional reverse mode fields |
| `summary.csv` | Extended with baseline_cf, target_prr, achieved_prr, dli_target_found |
| `points_dli.csv` | GIS point layer with DLI values for heatmap |

### Output Fields (Reverse Mode)

| Field | Unit | Description |
|-------|------|-------------|
| `baseline_cf_percent` | % | Baseline CF from unconstrained dynamic tracking |
| `target_prr_percent` | % | User-specified target PRR |
| `achieved_prr_percent` | % | Actual PRR achieved (CF/CF_ST × 100) |
| `dli_target_used` | mol/m²/day | DLI constraint value found by search |
| `dli_avg_achieved` | mol/m²/day | Mean DLI over simulation period |
| `drr_percent` | % | DLI Realization Ratio |

## Heatmap Visualization

The `heatmap.html` file reads `points_dli.csv` and displays an interactive DLI heatmap using ArcGIS API.

```bash
# Start local HTTP server from repository root
python -m http.server 8000

# Open in browser
# http://localhost:8000/AnalyzeData/heatmap.html
```

## Key Metrics

### Capacity Factor (CF)

```
CF = Σ P_gen(t) / (N_t × P_dc0) × 100
```

### Power Realization Ratio (PRR)

```
PRR = CF_CS / CF_ST × 100
```

Where CF_ST is the baseline capacity factor from unconstrained dynamic tracking.

### DLI Realization Ratio (DRR)

```
DRR = Σ DLI(d) / (N_d × DLI_target) × 100
```

## Configuration

Default parameters can be modified in `main.py` dataclasses:

- `LocationConfig`: Site coordinates and timezone
- `TrackingConfig`: Tracker strategy and mechanical limits
- `PVConfig`: Array geometry (width, spacing, height)
- `WeatherConfig`: PAR ratio, albedo, diffuse coefficients
- `PlantConfig`: Crop growth model parameters (RUE, GDD thresholds)
- `PowerConfig`: Module electrical parameters (Pdc0, temperature coefficients)
- `ModelControl`: DLI target and constraint settings

## NSRDB Data Format

Input CSV files should follow NSRDB TMY format:
- Row 1: Metadata headers (Source, Location ID, Latitude, Longitude, Time Zone, etc.)
- Row 2: Metadata values
- Row 3: Data column headers (Year, Month, Day, Hour, DNI, DHI, GHI, Temperature, etc.)
- Row 4+: Hourly weather data

Required columns: `Year`, `Month`, `Day`, `Hour`, `DNI`, `DHI`, `GHI`, `Temperature`

## License

Research use only. Contact the Agritrack Research Team for licensing inquiries.
