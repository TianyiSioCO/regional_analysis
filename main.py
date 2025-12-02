"""
Agrivoltaic Tracking Optimization System
=========================================

This module implements a nonlinear optimization model for single-axis tracker control
in agrivoltaic (agrophotovoltaic) systems. The system co-optimizes:
  - Solar power generation (maximizing energy yield)
  - Crop light requirements (Daily Light Integral constraints)

Key Features:
  - Bifacial PV module irradiance modeling (front + rear side)
  - Single-axis horizontal tracker with backtracking capability
  - DLI (Daily Light Integral) constraint for crop photosynthesis
  - Biomass accumulation model based on Radiation Use Efficiency (RUE)
  - PRR (Power Realization Ratio) and DRR (DLI Realization Ratio) metrics

References:
  - Agrivoltaic system design principles
  - NSRDB TMY (Typical Meteorological Year) data format
  - pvlib solar position algorithms

Author: Agritrack Research Team
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pvlib
from pyomo.environ import (
    ConcreteModel,
    Constraint,
    ConstraintList,
    Expression,
    Objective,
    Var,
    atan,  # type: ignore
    cos,
    maximize,
    sin,
    value,
)
from pyomo.opt import SolverFactory, TerminationCondition


# =============================================================================
# Configuration Data Classes
# =============================================================================

@dataclass
class LocationConfig:
    """
    Geographic location and temporal configuration.

    Attributes:
        latitude: Site latitude in decimal degrees (positive = North)
        longitude: Site longitude in decimal degrees (positive = East)
        timezone: IANA timezone string (e.g., 'Etc/GMT+5')
        start: Simulation start datetime (ISO format)
        end: Simulation end datetime (ISO format)
    """
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    timezone: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None

    def start_timestamp(self, tz: str) -> Optional[pd.Timestamp]:
        if self.start is None:
            return None
        return _to_timezone(pd.Timestamp(self.start), tz)

    def end_timestamp(self, tz: str) -> Optional[pd.Timestamp]:
        if self.end is None:
            return None
        return _to_timezone(pd.Timestamp(self.end), tz)


@dataclass
class TrackingConfig:
    """
    Single-axis tracker configuration.

    Attributes:
        strategy: Tracking mode - 'dynamic' (optimized), 'standard_tracking' (sun-following),
                  'antitracking' (perpendicular to sun), 'fixed_schedule', 'binary_select'
        start_hour: Start hour for fixed_schedule mode
        end_hour: End hour for fixed_schedule mode
        max_angle: Maximum tracker rotation angle [degrees] from horizontal
        delta_theta: Maximum angle change rate per timestep [degrees] (motor speed limit)
        backtrack: Enable backtracking to avoid inter-row shading
        initial_theta: Initial tracker tilt angle [radians]
    """
    strategy: str = "dynamic"
    start_hour: int = 10
    end_hour: int = 15
    max_angle: float = 60.0
    delta_theta: Optional[float] = None
    backtrack: bool = True
    initial_theta: float = 0.0


@dataclass
class PVConfig:
    """
    Photovoltaic array geometry configuration.

    Attributes:
        width: PV module/table width perpendicular to tracker axis [m]
        length: PV module/table length along tracker axis [m]
        height: Tracker axis height above ground [m]
        site_azimuth: Array azimuth angle [degrees], 0 = North, 90 = East
        interrow_spacing: Ground Cover Ratio (GCR) denominator - row pitch [m]
        intrarow_spacing: Gap between modules along the row [m]
        row_count: Number of tracker rows in the array
        surface_per_row: Number of module surfaces per row
        border: Edge buffer distance for shading calculations [m]
    """
    width: float = 2.0
    length: float = 20.0
    height: float = 2.0
    site_azimuth: float = 0.0
    interrow_spacing: float = 6.0
    intrarow_spacing: float = 0.0
    row_count: int = 14
    surface_per_row: int = 1
    border: float = 2.0


@dataclass
class WeatherConfig:
    """
    Weather and radiation parameters.

    Attributes:
        ratio_par: Fraction of solar radiation that is PAR (Photosynthetically Active Radiation)
                   Typical value ~0.45-0.50 of total solar spectrum
        qc_par: Quantum conversion factor [µmol photons / J]
                Converts PAR energy to photon flux for DLI calculation
        albedo: Ground surface reflectance (0-1), affects ground-reflected irradiance
        diffuse_fraction_coeff: Polynomial coefficients [a, b, c] for diffuse fraction model
                                f(θ) = a*θ^4 + b*θ^2 + c, where θ is tracker tilt [rad]
    """
    ratio_par: float = 0.5
    qc_par: float = 4.6
    albedo: float = 0.2
    diffuse_fraction_coeff: Sequence[float] = field(
        default_factory=lambda: [-0.01871471, 0.05827023, 0.6669363]
    )


@dataclass
class PlantConfig:
    """
    Crop growth model parameters (based on SIMPLE crop model).

    Attributes:
        rue: Radiation Use Efficiency [g DM / MJ PAR] - biomass produced per unit radiation
        base_temp: Base temperature for thermal time accumulation [°C]
        optimal_temp: Optimal growth temperature [°C]
        max_temp: Maximum temperature before heat stress [°C]
        extreme_temp: Lethal temperature threshold [°C]
        water_stress: Water stress coefficient (0-1)
        senescence_heat: Heat stress threshold for senescence [°C·day]
        senescence_water: Water stress threshold for senescence [days]
        temp_sum_plant: Total thermal time (GDD) for crop maturity [°C·day]
        harvest_index: Ratio of grain yield to total biomass
        temp_sum_leaf_area: Thermal time for maximum leaf area [°C·day]
        temp_sum_senescence: Thermal time before senescence begins [°C·day]
        canopy_max: Maximum fractional canopy cover (0-1)
        canopy_initial: Initial fractional canopy cover at emergence
        initial_biomass: Initial above-ground dry biomass [g/m²]
    """
    rue: float = 1.3
    base_temp: float = 4.0
    optimal_temp: float = 22.0
    max_temp: float = 34.0
    extreme_temp: float = 45.0
    water_stress: float = 0.4
    senescence_heat: float = 50.0
    senescence_water: float = 30.0
    temp_sum_plant: float = 2400.0
    harvest_index: float = 0.45
    temp_sum_leaf_area: float = 690.0
    temp_sum_senescence: float = 450.0
    canopy_max: float = 0.95
    canopy_initial: float = 0.001
    initial_biomass: float = 1.0


@dataclass
class PowerConfig:
    """
    PV module electrical and thermal parameters.

    Attributes:
        pdc0: Nameplate DC power capacity at STC [W]
        gamma_pdc: Temperature coefficient of power [1/°C], typically negative
        temperature_ref: Reference cell temperature at STC [°C]
        wind_speed_default: Default wind speed for cell temperature calculation [m/s]
        u_c: Combined heat loss coefficient (conductive + convective) [W/(m²·K)]
        u_v: Wind-dependent heat loss coefficient [W/(m²·K)/(m/s)]
        module_efficiency: Module conversion efficiency at STC (0-1)
        alpha_absorption: Solar absorptance of module surface (0-1)
        bifacial: Enable bifacial irradiance calculation
        bifaciality_factor: Rear-to-front efficiency ratio for bifacial modules (0-1)
    """
    pdc0: float = 5_000_000.0
    gamma_pdc: float = -0.003
    temperature_ref: float = 25.0
    wind_speed_default: float = 1.0
    u_c: float = 29.0
    u_v: float = 0.0
    module_efficiency: float = 0.1
    alpha_absorption: float = 0.9
    bifacial: bool = True
    bifaciality_factor: float = 0.8


@dataclass
class ModelControl:
    """
    Optimization objective and constraint settings.

    Attributes:
        criteria: Constraint type - 'dli' (Daily Light Integral) or 'biomass'
        dli_target: Target DLI for crop [mol/m²/day] - minimum light requirement
        biomass_requirement_ratio: Minimum biomass as fraction of reference (0-1)
    """
    criteria: str = "dli"
    dli_target: float = 30.0
    biomass_requirement_ratio: float = 0.6


@dataclass
class AgritrackConfig:
    """Master configuration container for all simulation parameters."""
    location: LocationConfig = field(default_factory=LocationConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    pv: PVConfig = field(default_factory=PVConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    plant: PlantConfig = field(default_factory=PlantConfig)
    power: PowerConfig = field(default_factory=PowerConfig)
    control: ModelControl = field(default_factory=ModelControl)


# =============================================================================
# Result Data Class
# =============================================================================

@dataclass
class AgritrackResult:
    """
    Optimization results container.

    Attributes:
        source: Input CSV file path
        metadata: NSRDB metadata from CSV header
        hourly_index: Datetime index for hourly timeseries
        power_w: Hourly DC power generation [W]
        dli_series: Hourly DLI values [mol/m²/day equivalent]
        cumulative_biomass_reference: Reference biomass without PV shading [g/m²]
        cumulative_biomass_model: Modeled biomass under PV [g/m²]
        solver_status: Pyomo solver status string
        solver_termination: Termination condition (optimal, infeasible, etc.)
        objective_value: Objective function value (total energy)
        capacity_factor: CF = actual_energy / (nameplate × hours) × 100 [%]
        prr: Power Realization Ratio (same as CF in forward mode) [%]

    Reverse Mode Fields (PRR → DLI search):
        baseline_cf: Baseline CF_ST from unconstrained dynamic tracking [%]
        target_prr: User-specified target PRR [%]
        achieved_prr: Actual PRR achieved = CF / CF_ST × 100 [%]
        dli_target_found: DLI constraint value found by binary search [mol/m²/day]
        dli_avg: Mean DLI achieved over simulation period [mol/m²/day]
        drr: DLI Realization Ratio = DLI_avg / DLI_target × 100 [%]
    """
    source: Path
    metadata: Dict[str, str]
    hourly_index: pd.DatetimeIndex
    power_w: np.ndarray
    dli_series: np.ndarray
    cumulative_biomass_reference: np.ndarray
    cumulative_biomass_model: Optional[np.ndarray]
    solver_status: str
    solver_termination: str
    objective_value: float
    capacity_factor: float
    prr: float
    # Reverse mode fields (PRR → DLI search)
    baseline_cf: Optional[float] = None
    target_prr: Optional[float] = None
    achieved_prr: Optional[float] = None
    dli_target_found: Optional[float] = None
    dli_avg: Optional[float] = None
    drr: Optional[float] = None


# =============================================================================
# Constants and Column Mappings
# =============================================================================

# NSRDB TMY CSV column name aliases for standardization
COLUMN_ALIASES: Dict[str, str] = {
    "source": "source",
    "location_id": "location_id",
    "city": "city",
    "state": "state",
    "country": "country",
    "latitude": "latitude",
    "longitude": "longitude",
    "time_zone": "time_zone",
    "elevation": "elevation",
    "local_time_zone": "local_time_zone",
    "year": "year",
    "month": "month",
    "day": "day",
    "hour": "hour",
    "minute": "minute",
    "dni": "dni",      # Direct Normal Irradiance [W/m²]
    "dhi": "dhi",      # Diffuse Horizontal Irradiance [W/m²]
    "ghi": "ghi",      # Global Horizontal Irradiance [W/m²]
    "temperature": "temperature",    # Ambient air temperature [°C]
    "wind_speed": "wind_speed",      # Wind speed at 10m height [m/s]
    "precipitation": "precipitation",
}


# =============================================================================
# Utility Functions
# =============================================================================

def _to_timezone(timestamp: pd.Timestamp, timezone: str) -> pd.Timestamp:
    """Convert timestamp to specified timezone."""
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(timezone)
    return timestamp.tz_convert(timezone)


def _offset_to_timezone(offset: int) -> str:
    """
    Convert UTC offset (e.g., -5) to IANA Etc/GMT timezone string.

    Note: Etc/GMT timezone signs are inverted from standard convention.
    Example: UTC-5 corresponds to Etc/GMT+5
    """
    if offset == 0:
        return "Etc/GMT"
    return f"Etc/GMT{-offset:+d}"


def _normalize_column(name: str) -> str:
    """Normalize column names to lowercase with underscores."""
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def _safe_float(value: Optional[str], fallback: float) -> float:
    """Safely convert string to float with fallback value."""
    try:
        if value is None or value == "":
            return fallback
        return float(value)
    except ValueError:
        return fallback


def _read_metadata(path: Path) -> Dict[str, str]:
    """
    Read NSRDB TMY file metadata from first two rows.

    NSRDB format: Row 1 = header names, Row 2 = values, Row 3+ = data
    """
    try:
        with path.open("r", encoding="utf-8") as file:
            header_line = file.readline().strip()
            value_line = file.readline().strip()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Cannot read file: {path}") from error

    metadata: Dict[str, str] = {}
    if header_line and value_line:
        headers = [_normalize_column(part) for part in header_line.split(",")]
        values = [part.strip() for part in value_line.split(",")]
        for key, val in zip_longest(headers, values, fillvalue=""):
            metadata[key] = val
    return metadata


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_weather_data(
    path: Path, location: LocationConfig
) -> Tuple[Dict[str, str], pd.DataFrame, LocationConfig]:
    """
    Load NSRDB TMY weather data with automatic metadata extraction.

    Parses NSRDB CSV format:
      - Row 1: Metadata headers (Source, Location ID, Latitude, etc.)
      - Row 2: Metadata values
      - Row 3: Data column headers (Year, Month, Day, Hour, DNI, DHI, GHI, etc.)
      - Row 4+: Hourly weather data

    Args:
        path: Path to NSRDB TMY CSV file
        location: Location configuration (auto-populated from metadata if empty)

    Returns:
        Tuple of (metadata dict, weather DataFrame, updated LocationConfig)
    """
    metadata = _read_metadata(path)
    raw = pd.read_csv(path, skiprows=2)
    normalized_columns = [_normalize_column(col) for col in raw.columns]
    raw.columns = normalized_columns
    rename_map = {col: COLUMN_ALIASES.get(col, col) for col in raw.columns}
    data = raw.rename(columns=rename_map)

    for column in ("year", "month", "day", "hour"):
        if column not in data.columns:
            raise ValueError(f"{path} missing required field: {column}")

    if "minute" not in data.columns:
        data["minute"] = 0

    numeric_candidates = [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "dni",
        "dhi",
        "ghi",
        "temperature",
        "wind_speed",
        "precipitation",
    ]
    for column in numeric_candidates:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    # Dynamically read timezone from metadata
    timezone = location.timezone
    if timezone is None:
        tz_offset_str = metadata.get("time_zone", "0")
        try:
            tz_offset = int(float(tz_offset_str))
        except ValueError:
            tz_offset = 0
        timezone = _offset_to_timezone(tz_offset)

    # Dynamically read latitude/longitude from metadata
    latitude = location.latitude
    if latitude is None:
        latitude = _safe_float(metadata.get("latitude"), 0.0)

    longitude = location.longitude
    if longitude is None:
        longitude = _safe_float(metadata.get("longitude"), 0.0)

    datetime_index = pd.to_datetime(
        data[["year", "month", "day", "hour", "minute"]],
        errors="coerce",
    )
    data = data.assign(datetime=datetime_index).dropna(subset=["datetime"])
    data = data.set_index("datetime").sort_index()
    data.index = data.index.tz_localize(timezone)

    # Automatically determine time range (defaults to full year)
    start_ts = location.start_timestamp(timezone)
    end_ts = location.end_timestamp(timezone)

    if start_ts is None or end_ts is None:
        # Use actual time range from data
        start_ts = data.index.min()
        end_ts = data.index.max()

    filtered = data.loc[(data.index >= start_ts) & (data.index <= end_ts)]

    required = ["dni", "dhi", "ghi", "temperature"]
    missing = [col for col in required if col not in filtered.columns]
    if missing:
        raise ValueError(f"{path} missing required fields for execution: {', '.join(missing)}")

    # Return updated LocationConfig
    updated_location = LocationConfig(
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
        start=str(start_ts),
        end=str(end_ts),
    )

    return metadata, filtered, updated_location


# =============================================================================
# Core Optimization Solver
# =============================================================================

def solve_for_weather(
    source: Path,
    metadata: Dict[str, str],
    weather_data: pd.DataFrame,
    config: AgritrackConfig,
    solver_name: str,
    solver_executable: Optional[str],
) -> AgritrackResult:
    """
    Solve the agrivoltaic tracking optimization problem.

    Optimization Model:
        Objective: Maximize total DC power generation over simulation period
        Decision Variable: θ(t) - tracker tilt angle at each timestep [radians]
        Subject to:
            - |θ(t)| ≤ θ_max (maximum rotation angle)
            - DLI constraint: mean(DLI) ≥ DLI_target (crop light requirement)
            - Optional: biomass constraint, angle rate limits, backtracking

    Irradiance Model (Bifacial):
        Front: G_front = DNI·cos(AOI) + DHI·(1+cos(θ+β_L))/2 + ρ·G_ground·(1-cos(θ-β_S))/2
        Back:  G_back = DHI·(1-cos(θ-β_S))/2 + ρ·G_ground·(1+cos(θ+β_L))/2
        where β_L, β_S are blockage angles from adjacent rows

    Power Model:
        P_dc = P_dc0 × (G_incident/1000) × (1 + γ × (T_cell - T_ref))

    Args:
        source: Input CSV file path
        metadata: NSRDB metadata dictionary
        weather_data: Hourly weather DataFrame with DNI, DHI, GHI, Temperature
        config: Complete configuration object
        solver_name: Pyomo solver name (e.g., 'ipopt')
        solver_executable: Optional path to solver executable

    Returns:
        AgritrackResult containing optimized power, DLI series, and metrics
    """
    location = config.location
    tracking = config.tracking
    pv_cfg = config.pv
    weather_cfg = config.weather
    plant_cfg = config.plant
    power_cfg = config.power
    control = config.control

    hourly_timesteps = weather_data.index
    if len(hourly_timesteps) < 2:
        raise ValueError("Insufficient timesteps: need at least 2 data points")
    time_interval = (hourly_timesteps[1] - hourly_timesteps[0]).total_seconds()

    # -------------------------------------------------------------------------
    # Step 1: Calculate solar position using pvlib
    # θ_e = solar elevation angle, φ_a = solar azimuth angle
    # -------------------------------------------------------------------------
    latitude = _safe_float(metadata.get("latitude"), location.latitude)
    longitude = _safe_float(metadata.get("longitude"), location.longitude)
    site_location = pvlib.location.Location(
        latitude=latitude,
        longitude=longitude,
        tz=location.timezone,
    )
    solar_angles = site_location.get_solarposition(hourly_timesteps)
    theta_e_deg = solar_angles["apparent_elevation"].astype(float).values
    phi_a_deg = solar_angles["azimuth"].astype(float).values

    # Extract irradiance components from weather data
    dni = weather_data["dni"].astype(float).values  # Direct Normal Irradiance [W/m²]
    dhi = weather_data["dhi"].astype(float).values  # Diffuse Horizontal Irradiance [W/m²]
    ghi = weather_data["ghi"].astype(float).values  # Global Horizontal Irradiance [W/m²]
    air_temp = weather_data["temperature"].astype(float).values  # Ambient temperature [°C]

    # -------------------------------------------------------------------------
    # Step 2: Precompute trigonometric values for solar geometry
    # φ_r = relative azimuth angle (sun azimuth - array azimuth)
    # -------------------------------------------------------------------------
    theta_e_deg[ghi > 0] = np.maximum(theta_e_deg[ghi > 0], 5.0)  # Min elevation when sun is up
    theta_e_deg[ghi == 0] = 1e-6  # Avoid division by zero at night
    phi_r = np.deg2rad(phi_a_deg - pv_cfg.site_azimuth)  # Relative azimuth [rad]
    theta_e = np.deg2rad(theta_e_deg)  # Solar elevation [rad]

    # Precompute trigonometric functions for efficiency
    sin_phi_r = np.sin(phi_r)
    cos_phi_r = np.cos(phi_r)
    tan_theta_e = np.tan(theta_e)
    sin_theta_e = np.sin(theta_e)
    cos_theta_e = np.cos(theta_e)

    # -------------------------------------------------------------------------
    # Step 3: Calculate reference tracking angles
    # Optimal angle: tracker perpendicular to sun rays (maximizes DNI capture)
    # θ_opt = arctan(cos(θ_e)·sin(φ_r) / sin(θ_e))
    # -------------------------------------------------------------------------
    a, b, c = weather_cfg.diffuse_fraction_coeff
    optimal_angle = np.arctan((cos_theta_e * sin_phi_r) / sin_theta_e)
    theta_z_xz = np.arctan((cos_theta_e * np.abs(sin_phi_r)) / sin_theta_e)
    max_tilt_rad = np.radians(tracking.max_angle)

    # Standard tracking: follow optimal angle within mechanical limits
    constrained_tracking = np.clip(optimal_angle, -max_tilt_rad, max_tilt_rad)

    # Anti-tracking: perpendicular to optimal (maximizes light to ground)
    constrained_antitracking = np.clip(
        optimal_angle - np.sign(optimal_angle) * np.pi / 2,
        -max_tilt_rad,
        max_tilt_rad,
    )

    sun_up = np.heaviside(theta_e, 1)  # Binary mask: 1 when sun is up

    # -------------------------------------------------------------------------
    # Step 4: Apply backtracking to avoid inter-row shading
    # When projected shadow width > row spacing, reduce tilt angle
    # -------------------------------------------------------------------------
    panel_width_shadow_tracking = (
        (sin_phi_r / np.tan(theta_e) * np.sin(constrained_tracking) + np.cos(constrained_tracking))
        * pv_cfg.width
        * sun_up
    )
    angle_tracking = constrained_tracking.copy()
    angle_antitracking = constrained_antitracking.copy()

    # Backtracking adjustment when shading occurs
    for idx in range(len(theta_e)):
        if panel_width_shadow_tracking[idx] > pv_cfg.interrow_spacing:
            dtheta = np.arccos(
                pv_cfg.interrow_spacing / pv_cfg.width * np.cos(theta_z_xz[idx])
            )
            if optimal_angle[idx] >= 0:
                angle_tracking[idx] = optimal_angle[idx] - dtheta
            else:
                angle_tracking[idx] = optimal_angle[idx] + dtheta

    angle_tracking *= sun_up
    angle_antitracking *= sun_up

    # -------------------------------------------------------------------------
    # Step 5: Crop Growth Model (SIMPLE model adaptation)
    # Calculate thermal time (Growing Degree Days), canopy cover, and stress factors
    # -------------------------------------------------------------------------
    steps_per_day = 86400 / time_interval
    n_days = int(len(hourly_timesteps) / steps_per_day)

    # Aggregate daily temperature statistics
    daily_temp_stats = weather_data.resample("1D").agg(["min", "max", "mean"])[
        "temperature"
    ]
    daily_min_temp = daily_temp_stats["min"].values
    daily_max_temp = daily_temp_stats["max"].values
    daily_mean_temp = daily_temp_stats["mean"].values

    # Daily solar radiation [MJ/m²/day] for biomass calculation
    daily_srads = (
        weather_data.resample("1D").agg(["mean"])["ghi"]["mean"].values * 86400 * 1e-6
    )
    n_days = min(n_days, len(daily_srads))
    daily_srads = daily_srads[:n_days]
    daily_min_temp = daily_min_temp[:n_days]
    daily_max_temp = daily_max_temp[:n_days]
    daily_mean_temp = daily_mean_temp[:n_days]

    # Thermal time (GDD) accumulation: Σmax(T_mean - T_base, 0)
    d_thermal_time = np.maximum(daily_mean_temp - plant_cfg.base_temp, 0.0)
    thermal_time = np.zeros(n_days + 1)
    thermal_time[0] = 0.0
    for day in range(1, n_days + 1):
        thermal_time[day] = thermal_time[day - 1] + d_thermal_time[day - 1]
        if thermal_time[day] > plant_cfg.temp_sum_plant:
            n_days = day
            break

    # Temperature stress factor: linear response between base and optimal temp
    temp_growth = np.clip(
        (daily_mean_temp - plant_cfg.base_temp)
        / (plant_cfg.optimal_temp - plant_cfg.base_temp),
        0,
        1,
    )

    # Heat stress factor: linear decline above max temp
    heat_growth = np.clip(
        (plant_cfg.extreme_temp - daily_max_temp)
        / (plant_cfg.extreme_temp - plant_cfg.max_temp),
        0,
        1,
    )

    # Canopy cover development: sigmoid growth with senescence
    # Uses logistic function based on thermal time
    canopy_growth = np.zeros(n_days + 1)
    canopy_growth[0] = plant_cfg.canopy_initial
    canopy_growth[1:] = np.minimum(
        plant_cfg.canopy_max
        / (1 + np.exp(-0.01 * (thermal_time[1:] - plant_cfg.temp_sum_leaf_area))),
        plant_cfg.canopy_max
        / (
            1
            + np.exp(0.01 * (thermal_time[1:] - plant_cfg.temp_sum_plant + plant_cfg.temp_sum_senescence))
        ),
    )

    # Reference biomass: crop yield without PV shading (full sunlight)
    # Biomass = RUE × Radiation × Canopy × TempFactor × HeatFactor
    biomass_reference = plant_cfg.initial_biomass + sum(
        10
        * plant_cfg.rue
        * daily_srads
        * canopy_growth[:-1]
        * temp_growth[: len(daily_srads)]
        * heat_growth[: len(daily_srads)]
    )
    cumulative_biomass_reference = np.zeros(n_days + 1)
    cumulative_biomass_reference[0] = plant_cfg.initial_biomass
    for day in range(1, n_days + 1):
        cumulative_biomass_reference[day] = cumulative_biomass_reference[day - 1] + (
            10
            * plant_cfg.rue
            * daily_srads[day - 1]
            * canopy_growth[day - 1]
            * temp_growth[day - 1]
            * heat_growth[day - 1]
        )

    # -------------------------------------------------------------------------
    # Step 6: Build Pyomo Optimization Model
    # Decision variable: θ(t) = tracker tilt angle at each timestep
    # -------------------------------------------------------------------------
    model = ConcreteModel()

    penalty_weight = 0.0

    # Define tracker angle variable based on tracking strategy
    if tracking.strategy in {"dynamic", "standard_tracking", "antitracking", "fixed_schedule"}:
        # Continuous variable for tracker angle [radians]
        model.theta = Var(
            range(len(hourly_timesteps)),
            bounds=(-np.radians(tracking.max_angle), np.radians(tracking.max_angle)),
            initialize=tracking.initial_theta,
        )
        # Fix angles for non-optimized strategies
        if tracking.strategy == "standard_tracking":
            for idx in range(len(hourly_timesteps)):
                model.theta[idx].fix(angle_tracking[idx])
        elif tracking.strategy == "antitracking":
            for idx in range(len(hourly_timesteps)):
                model.theta[idx].fix(angle_antitracking[idx])
        elif tracking.strategy == "fixed_schedule":
            for idx in range(len(hourly_timesteps)):
                if tracking.start_hour <= hourly_timesteps[idx].hour <= tracking.end_hour:
                    model.theta[idx].fix(angle_tracking[idx])
                else:
                    model.theta[idx].fix(angle_antitracking[idx])
    elif tracking.strategy == "binary_select":
        # Binary selection between tracking and anti-tracking (relaxed to continuous)
        model.switch = Var(range(len(hourly_timesteps)), bounds=(0, 1), initialize=1.0)
        model.theta = Expression(
            range(len(hourly_timesteps)),
            rule=lambda m, t: angle_tracking[t] * m.switch[t] + angle_antitracking[t] * (1 - m.switch[t]),
        )
        # Penalty term to encourage binary solutions
        penalty_expr = sum(
            model.switch[t] * (1 - model.switch[t]) for t in range(len(hourly_timesteps))
        )
        model.penalty = Expression(expr=penalty_expr)
        penalty_weight = 1e6
    else:
        raise ValueError(f"Unknown tracking strategy: {tracking.strategy}")

    # -------------------------------------------------------------------------
    # Step 7: Ground Irradiance Model
    # Calculate irradiance reaching the ground between panel rows
    # -------------------------------------------------------------------------

    # Projected shadow width on ground [m]
    def panel_width_shadow_expr(mdl, t):
        return (
            (sin_phi_r[t] / tan_theta_e[t] * sin(mdl.theta[t]) + cos(mdl.theta[t])) * pv_cfg.width
        )

    model.panel_width_shadow = Expression(
        range(len(hourly_timesteps)), rule=panel_width_shadow_expr
    )

    # Shadow constraints: ensure valid geometry and backtracking limits
    model.shadow_constraints = ConstraintList()
    for idx in range(len(hourly_timesteps)):
        model.shadow_constraints.add(model.panel_width_shadow[idx] >= 0.0)
        if tracking.backtrack:
            model.shadow_constraints.add(model.panel_width_shadow[idx] <= pv_cfg.interrow_spacing)

    # Diffuse fraction reaching ground: polynomial function of tilt angle
    # f(θ) = a·θ⁴ + b·θ² + c (empirical model)
    def diffuse_fraction_expr(mdl, t):
        return a * mdl.theta[t] ** 4 + b * mdl.theta[t] ** 2 + c

    model.diffuse_fraction = Expression(
        range(len(hourly_timesteps)),
        rule=diffuse_fraction_expr,
    )

    # Ground irradiance: direct beam (with shading) + diffuse
    # G_ground = (1 - shadow_width/pitch) × DNI × sin(θ_e) + DHI × f_diffuse
    def ground_radiation_expr(mdl, t):
        return (
            (1 - model.panel_width_shadow[t] / pv_cfg.interrow_spacing)
            * sin(theta_e[t])
            * dni[t]
            + dhi[t] * model.diffuse_fraction[t]
        )

    model.ground_radiation = Expression(range(len(hourly_timesteps)), rule=ground_radiation_expr)

    # Ground radiation energy for plant model [MJ/m²]
    def ground_radiation_plant_expr(mdl, t):
        return model.ground_radiation[t] * time_interval / 1e6

    model.ground_radiation_plant = Expression(
        range(len(hourly_timesteps)), rule=ground_radiation_plant_expr
    )

    # -------------------------------------------------------------------------
    # Step 8: Crop Constraint (DLI or Biomass)
    # Ensures sufficient light reaches crops beneath panels
    # -------------------------------------------------------------------------
    if control.criteria == "biomass":
        # Biomass constraint: final biomass ≥ requirement ratio × reference
        model.daily_radiation = Expression(
            range(n_days),
            rule=lambda mdl, day: _daily_radiation_average(mdl, day, steps_per_day, time_interval),
        )
        model.cumulative_biomass = Var(range(n_days + 1), initialize=plant_cfg.initial_biomass)
        model.cumulative_biomass_constraints = ConstraintList()
        model.cumulative_biomass_constraints.add(model.cumulative_biomass[0] == plant_cfg.initial_biomass)
        for day in range(n_days):
            model.cumulative_biomass_constraints.add(
                model.cumulative_biomass[day + 1]
                == model.cumulative_biomass[day]
                + 10.0
                * plant_cfg.rue
                * model.daily_radiation[day]
                * canopy_growth[day]
                * temp_growth[day]
                * heat_growth[day]
            )
        model.biomass_requirement = Constraint(
            expr=model.cumulative_biomass[n_days]
            >= biomass_reference * control.biomass_requirement_ratio
        )
    elif control.criteria == "dli":
        # DLI constraint: mean(DLI) ≥ DLI_target
        # DLI [mol/m²/day] = G_ground × PAR_ratio × quantum_conversion × seconds_per_day / 1e6
        model.dli_expr = Expression(
            range(len(hourly_timesteps)),
            rule=lambda mdl, t: mdl.ground_radiation[t]
            * weather_cfg.ratio_par
            * weather_cfg.qc_par
            * 24
            * 3600
            / 1e6,
        )
        model.dli_constraint = Constraint(
            expr=sum(model.dli_expr[t] for t in range(len(hourly_timesteps))) / len(hourly_timesteps)
            >= control.dli_target
        )
    else:
        raise ValueError(f"Unknown crop constraint type: {control.criteria}")

    # Motor speed constraint: limit angle change rate between timesteps
    if tracking.delta_theta is not None:
        model.angle_change_constraints = ConstraintList()
        for idx in range(len(hourly_timesteps) - 1):
            step_limit = tracking.delta_theta / 180 * np.pi
            model.angle_change_constraints.add(model.theta[idx + 1] - model.theta[idx] <= step_limit)
            model.angle_change_constraints.add(model.theta[idx + 1] - model.theta[idx] >= -step_limit)

    # -------------------------------------------------------------------------
    # Step 9: Bifacial PV Irradiance Model
    # Calculate front and back surface irradiance for bifacial modules
    # -------------------------------------------------------------------------

    # Cosine of Angle of Incidence (AOI) for direct beam on tilted surface
    # cos(AOI) = cos(θ)·sin(θ_e) + sin(θ)·cos(θ_e)·sin(φ_r)
    def cos_aoi_expr(mdl, t):
        return cos(mdl.theta[t]) * sin(theta_e[t]) + sin(mdl.theta[t]) * cos(theta_e[t]) * sin_phi_r[t]

    model.cos_aoi = Expression(range(len(hourly_timesteps)), rule=cos_aoi_expr)

    # Blockage angles from adjacent rows (view factor geometry)
    # β_large: angle to far edge of adjacent row
    # β_small: angle to near edge of adjacent row
    def blockage_angle_large_expr(mdl, t):
        return atan(
            pv_cfg.width / 2 * sin(mdl.theta[t])
            / (pv_cfg.interrow_spacing - pv_cfg.width / 2 * cos(mdl.theta[t]))
        )

    def blockage_angle_small_expr(mdl, t):
        return atan(
            pv_cfg.width / 2 * sin(mdl.theta[t])
            / (pv_cfg.interrow_spacing + pv_cfg.width / 2 * cos(mdl.theta[t]))
        )

    model.blockage_angle_large = Expression(range(len(hourly_timesteps)), rule=blockage_angle_large_expr)
    model.blockage_angle_small = Expression(range(len(hourly_timesteps)), rule=blockage_angle_small_expr)

    # Front surface irradiance (faces sun):
    # G_front = DNI·cos(AOI) + DHI·sky_view_factor + ρ·G_ground·ground_view_factor
    def front_surface_irradiance_expr(mdl, t):
        return (
            mdl.cos_aoi[t] * dni[t]
            + dhi[t] * (cos(mdl.theta[t] + mdl.blockage_angle_large[t]) + 1) / 2
            + weather_cfg.albedo * mdl.ground_radiation[t] * (1 - cos(mdl.theta[t] - mdl.blockage_angle_small[t])) / 2
        )

    model.front_surface_irradiance = Expression(
        range(len(hourly_timesteps)), rule=front_surface_irradiance_expr
    )

    # Back surface irradiance (faces ground):
    def back_surface_irradiance_expr(mdl, t):
        return (
            dhi[t] * (1 - cos(mdl.theta[t] - mdl.blockage_angle_small[t])) / 2
            + weather_cfg.albedo * mdl.ground_radiation[t] * (cos(mdl.theta[t] + mdl.blockage_angle_large[t]) + 1) / 2
        )

    # G_back = DHI·(1-cos(θ-β_S))/2 + ρ·G_ground·(1+cos(θ+β_L))/2
    model.back_surface_irradiance = Expression(
        range(len(hourly_timesteps)), rule=back_surface_irradiance_expr
    )

    # Total incident irradiance: front + bifaciality_factor × back
    if power_cfg.bifacial:
        model.incident_irradiance = Expression(
            range(len(hourly_timesteps)),
            rule=lambda mdl, t: mdl.front_surface_irradiance[t]
            + mdl.back_surface_irradiance[t] * power_cfg.bifaciality_factor,
        )
    else:
        model.incident_irradiance = Expression(
            range(len(hourly_timesteps)),
            rule=lambda mdl, t: mdl.front_surface_irradiance[t],
        )

    # -------------------------------------------------------------------------
    # Step 10: Power Generation Model
    # Cell temperature and DC power output calculation
    # -------------------------------------------------------------------------

    # Cell temperature model (Faiman model):
    # T_cell = T_ambient + (G × α × (1-η)) / (U_c + U_v × v_wind)
    def cell_temp_expr(mdl, t):
        heat_input = (
            mdl.front_surface_irradiance[t] + mdl.back_surface_irradiance[t]
        ) * power_cfg.alpha_absorption * (1 - power_cfg.module_efficiency)
        temp_difference = heat_input / (power_cfg.u_c + power_cfg.u_v * power_cfg.wind_speed_default)
        return temp_difference + air_temp[t]

    model.cell_temp = Expression(range(len(hourly_timesteps)), rule=cell_temp_expr)

    # DC power output with temperature derating:
    # P_dc = P_dc0 × (G/1000) × (1 + γ × (T_cell - T_ref))
    model.power_generation = Expression(
        range(len(hourly_timesteps)),
        rule=lambda mdl, t: power_cfg.pdc0
        * mdl.incident_irradiance[t]
        / 1000
        * (1 + power_cfg.gamma_pdc * (mdl.cell_temp[t] - power_cfg.temperature_ref)),
    )

    # -------------------------------------------------------------------------
    # Step 11: Objective Function and Solve
    # Maximize total energy generation subject to constraints
    # -------------------------------------------------------------------------
    def objective_rule(mdl):
        total_power = sum(mdl.power_generation[t] for t in range(len(hourly_timesteps)))
        if tracking.strategy == "binary_select":
            return total_power - penalty_weight * model.penalty
        return total_power

    model.objective = Objective(rule=objective_rule, sense=maximize)

    # Solve with nonlinear solver (IPOPT)
    solver = SolverFactory(solver_name, executable=solver_executable) if solver_executable else SolverFactory(solver_name)
    if solver is None:
        raise RuntimeError(f"Cannot create solver: {solver_name}")
    results = solver.solve(model, tee=False)
    termination = str(results.solver.termination_condition)
    status = str(results.solver.status)

    if results.solver.termination_condition not in (
        TerminationCondition.optimal,
        TerminationCondition.locallyOptimal,
    ):
        raise RuntimeError(f"Solver did not find optimal solution: {termination}")

    power_values = np.array(
        [float(value(model.power_generation[idx])) for idx in range(len(hourly_timesteps))]
    )
    dli_series = np.array(
        [
            float(
                value(model.ground_radiation[idx])
                * weather_cfg.ratio_par
                * weather_cfg.qc_par
                * 24
                * 3600
                / 1e6
            )
            for idx in range(len(hourly_timesteps))
        ]
    )
    model_cumulative = None
    if hasattr(model, "cumulative_biomass"):
        model_cumulative = np.array(
            [float(value(model.cumulative_biomass[idx])) for idx in range(n_days + 1)]
        )

    total_power = float(np.sum(power_values))
    capacity_factor = (
        total_power / (len(power_values) * power_cfg.pdc0) * 100 if len(power_values) else 0.0
    )
    prr = capacity_factor

    return AgritrackResult(
        source=source,
        metadata=metadata,
        hourly_index=hourly_timesteps,
        power_w=power_values,
        dli_series=dli_series,
        cumulative_biomass_reference=cumulative_biomass_reference,
        cumulative_biomass_model=model_cumulative,
        solver_status=status,
        solver_termination=termination,
        objective_value=float(value(model.objective)),
        capacity_factor=capacity_factor,
        prr=prr,
    )


def _daily_radiation_average(model: ConcreteModel, day: int, steps_per_day: float, time_interval: float):
    start_idx = round(day * steps_per_day)
    end_idx = round((day + 1) * steps_per_day)
    duration = (end_idx - start_idx) * time_interval
    return sum(model.ground_radiation_plant[t] for t in range(start_idx, end_idx)) / (duration / 86400)


def compute_baseline_cf(
    source: Path,
    metadata: Dict[str, str],
    weather_data: pd.DataFrame,
    config: AgritrackConfig,
    solver_name: str,
    solver_executable: Optional[str],
) -> float:
    """Compute baseline CF_ST: unconstrained dynamic tracking without DLI constraints.

    Args:
        source: Path to CSV file
        metadata: Metadata dictionary
        weather_data: Weather data DataFrame
        config: Configuration (temporarily modified to unconstrained mode)
        solver_name: Solver name
        solver_executable: Optional path to solver executable

    Returns:
        Baseline capacity factor CF_ST (percentage)
    """
    # Create configuration copy with unconstrained dynamic tracking
    import copy
    baseline_config = copy.deepcopy(config)
    baseline_config.tracking.strategy = "dynamic"
    # Set very low DLI target to effectively disable constraint
    baseline_config.control.dli_target = 0.0

    try:
        result = solve_for_weather(
            source, metadata, weather_data, baseline_config, solver_name, solver_executable
        )
        return result.capacity_factor
    except RuntimeError as e:
        print(f"Warning: Solver failed during baseline CF calculation: {e}")
        return 0.0


def search_dli_for_target_prr(
    source: Path,
    metadata: Dict[str, str],
    weather_data: pd.DataFrame,
    config: AgritrackConfig,
    solver_name: str,
    solver_executable: Optional[str],
    baseline_cf: float,
    target_prr: float,
    dli_min: float,
    dli_max: float,
    tolerance: float,
    max_iter: int,
) -> AgritrackResult:
    """Binary search for DLI value that achieves target PRR.

    PRR = CF_CS / CF_ST × 100
    Objective: Find DLI such that PRR ≈ target_prr

    Args:
        source: Path to CSV file
        metadata: Metadata dictionary
        weather_data: Weather data DataFrame
        config: Base configuration
        solver_name: Solver name
        solver_executable: Optional path to solver executable
        baseline_cf: Baseline CF_ST
        target_prr: Target PRR percentage
        dli_min: Lower bound for DLI search
        dli_max: Upper bound for DLI search
        tolerance: PRR error tolerance (percentage points)
        max_iter: Maximum number of search iterations

    Returns:
        AgritrackResult containing search results
    """
    import copy

    # Calculate target CF
    target_cf = target_prr * baseline_cf / 100.0
    print(f"  Baseline CF_ST: {baseline_cf:.2f}%")
    print(f"  Target PRR: {target_prr:.1f}% -> Target CF: {target_cf:.2f}%")
    print(f"  DLI search range: [{dli_min}, {dli_max}] mol/m²/day")

    low, high = dli_min, dli_max
    best_result: Optional[AgritrackResult] = None
    best_prr_diff = float("inf")
    best_dli = dli_min

    for iteration in range(max_iter):
        mid_dli = (low + high) / 2.0

        # 使用当前 DLI 目标运行优化
        test_config = copy.deepcopy(config)
        test_config.control.dli_target = mid_dli

        try:
            result = solve_for_weather(
                source, metadata, weather_data, test_config, solver_name, solver_executable
            )
            current_cf = result.capacity_factor
            current_prr = (current_cf / baseline_cf) * 100.0 if baseline_cf > 0 else 0.0
            prr_diff = abs(current_prr - target_prr)

            print(f"  Iteration {iteration + 1}: DLI={mid_dli:.2f} -> CF={current_cf:.2f}%, PRR={current_prr:.2f}%")

            # Record best result
            if prr_diff < best_prr_diff:
                best_prr_diff = prr_diff
                best_result = result
                best_dli = mid_dli

            # Check if tolerance is reached
            if prr_diff <= tolerance:
                print(f"  Converged! Found DLI={mid_dli:.2f} achieving PRR={current_prr:.2f}% (target {target_prr}%)")
                break

            # Binary search logic:
            # Higher DLI -> stronger power constraint -> lower CF -> lower PRR
            # If PRR is too high, increase DLI (stronger constraint)
            if current_prr > target_prr:
                low = mid_dli  # PRR too high, raise DLI lower bound
            else:
                high = mid_dli  # PRR too low, reduce DLI upper bound

        except RuntimeError as e:
            print(f"  Iteration {iteration + 1}: DLI={mid_dli:.2f} -> Solver failed: {e}")
            # If solver fails, DLI constraint is too strong, reduce DLI
            high = mid_dli

    if best_result is None:
        raise RuntimeError(f"Unable to find DLI value satisfying target PRR={target_prr}%")

    # Calculate DLI-related metrics
    dli_avg = float(np.mean(best_result.dli_series))
    achieved_prr = (best_result.capacity_factor / baseline_cf) * 100.0 if baseline_cf > 0 else 0.0
    # DRR = actual average DLI / target DLI × 100
    drr = (dli_avg / best_dli) * 100.0 if best_dli > 0 else 0.0

    # Update result object
    best_result.baseline_cf = baseline_cf
    best_result.target_prr = target_prr
    best_result.achieved_prr = achieved_prr
    best_result.dli_target_found = best_dli
    best_result.dli_avg = dli_avg
    best_result.drr = drr

    return best_result


def write_outputs(
    result: AgritrackResult,
    output_dir: Path,
    reverse_mode: bool = False,
    dli_target_used: Optional[float] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Write output files.

    Args:
        result: Computation results
        output_dir: Output directory
        reverse_mode: Whether in reverse mode (PRR -> DLI)
        dli_target_used: DLI target value used in forward mode

    Returns:
        Tuple of timeseries file path and point record dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = result.source.stem

    # Simplified timeseries output: keep only core fields
    df = pd.DataFrame(
        {
            "datetime": result.hourly_index,
            "power_w": result.power_w,
            "dli": result.dli_series,
        }
    )

    timeseries_path = output_dir / f"{base_name}_timeseries.csv"
    df.to_csv(timeseries_path, index=False)

    latitude = _safe_float(result.metadata.get("latitude"), float("nan"))
    longitude = _safe_float(result.metadata.get("longitude"), float("nan"))

    # Calculate average DLI
    dli_avg = float(np.mean(result.dli_series)) if result.dli_series is not None else 0.0

    if reverse_mode and result.baseline_cf is not None:
        # Reverse mode (PRR -> DLI): output DLI map fields
        drr = result.drr if result.drr is not None else 0.0
        point_record = {
            "latitude": latitude,
            "longitude": longitude,
            "dli_target_found": result.dli_target_found,
            "dli_avg": result.dli_avg,
            "drr_percent": drr,
        }
    else:
        # Forward mode (DLI -> PRR): output Power map fields
        # DRR = actual average DLI / target DLI × 100
        dli_target = dli_target_used if dli_target_used else 30.0
        drr = (dli_avg / dli_target) * 100.0 if dli_target > 0 else 0.0
        point_record = {
            "latitude": latitude,
            "longitude": longitude,
            "prr_percent": result.prr,
            "dli_avg": dli_avg,
            "drr_percent": drr,
        }

    return str(timeseries_path), point_record


def find_data_files(data_dir: Path) -> list:
    """
    Automatically scan for CSV files in subdirectories under data directory.

    Directory structure example:
        data/
        ├── 1/
        │   └── 1077174_42.49_-79.46_tmy-2024.csv
        ├── 2/
        │   └── 1080822_42.49_-79.30_tmy-2024.csv
        └── ...

    Returns:
        List of CSV file paths
    """
    csv_files = []
    if not data_dir.exists():
        return csv_files

    # Iterate through subdirectories
    for subdir in sorted(data_dir.iterdir()):
        if subdir.is_dir():
            # Find CSV files in subdirectory (exclude non-TMY data files)
            for csv_file in subdir.glob("*_tmy*.csv"):
                csv_files.append(str(csv_file))

    return csv_files


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch computation of agrivoltaic tracking strategy results")

    # Compute default input files: automatically scan data subdirectories
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    default_inputs = find_data_files(data_dir)
    if not default_inputs:
        # If no subdirectory files found, fall back to legacy default
        default_inputs = [str(script_dir / "data" / "data1.csv")]

    parser.add_argument(
        "--inputs",
        nargs="+",
        default=default_inputs,
        help="CSV paths to process, defaults to auto-scanning TMY files in data/ subdirectories",
    )
    parser.add_argument(
        "--criteria",
        choices=["dli", "biomass"],
        default="dli",
        help="Plant constraint type, supports DLI or Biomass",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("AnalyzeData/output")),
        help="Output directory for results",
    )
    parser.add_argument(
        "--solver",
        default="ipopt",
        help="Pyomo solver name, default: ipopt",
    )
    parser.add_argument(
        "--solver-executable",
        default=None,
        help="Optional: path to solver executable",
    )
    # Reverse mode parameters: input PRR% to search for DLI (reverse mode enabled by default)
    parser.add_argument(
        "--target-prr",
        type=float,
        default=90.0,
        help="Target PRR percentage (reverse mode), default 90%%. Set to 0 to switch to forward mode",
    )
    parser.add_argument(
        "--forward",
        action="store_true",
        help="Use forward mode (input DLI -> compute PRR), overrides --target-prr",
    )
    parser.add_argument(
        "--dli-target",
        type=float,
        default=30.0,
        help="DLI target value for forward mode (mol/m²/day), default 30",
    )
    parser.add_argument(
        "--dli-min",
        type=float,
        default=15.0,
        help="Lower bound for DLI search (mol/m²/day), default 15",
    )
    parser.add_argument(
        "--dli-max",
        type=float,
        default=50.0,
        help="Upper bound for DLI search (mol/m²/day), default 50",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.5,
        help="PRR error tolerance (percentage points), default 0.5",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=20,
        help="Maximum search iterations, default 20",
    )

    args = parser.parse_args()

    # If --forward is specified, disable reverse mode
    if args.forward:
        args.target_prr = None
    # If target_prr is 0, also treat as disabled reverse mode
    elif args.target_prr == 0:
        args.target_prr = None

    return args


def main() -> None:
    args = parse_arguments()
    config = AgritrackConfig()
    config.control.criteria = args.criteria
    config.control.dli_target = args.dli_target  # Set DLI target from command line arguments
    output_dir = Path(args.output_dir)
    summaries = []
    point_records = []

    # Determine execution mode
    reverse_mode = args.target_prr is not None
    if reverse_mode:
        print(f"=== Reverse Mode: Input PRR={args.target_prr}% -> Search DLI ===")
        print(f"  DLI search range: [{args.dli_min}, {args.dli_max}] mol/m²/day")
        print(f"  PRR tolerance: {args.tolerance}%")
    else:
        print(f"=== Forward Mode: Input DLI={args.dli_target} -> Compute PRR ===")

    for idx, input_path in enumerate(args.inputs, 1):
        csv_path = Path(input_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Input file not found: {csv_path}")
        # Use new LocationConfig for each file to avoid location info reuse
        metadata, weather_data, updated_location = load_weather_data(csv_path, LocationConfig())
        config.location = updated_location
        print(f"\n[{idx}/{len(args.inputs)}] Loaded file: {csv_path}")
        print(f"  Location: ({updated_location.latitude}, {updated_location.longitude})")
        print(f"  Timezone: {updated_location.timezone}")
        print(f"  Time range: {updated_location.start} ~ {updated_location.end}")

        if reverse_mode:
            # Reverse mode: first compute baseline CF_ST, then search for DLI
            print("  Computing baseline CF_ST (unconstrained dynamic tracking)...")
            baseline_cf = compute_baseline_cf(
                csv_path, metadata, weather_data, config, args.solver, args.solver_executable
            )
            print(f"  Baseline CF_ST: {baseline_cf:.2f}%")

            print("  Searching for DLI that achieves target PRR...")
            result = search_dli_for_target_prr(
                csv_path, metadata, weather_data, config,
                args.solver, args.solver_executable,
                baseline_cf, args.target_prr,
                args.dli_min, args.dli_max,
                args.tolerance, args.max_iter
            )

            timeseries_path, point_record = write_outputs(result, output_dir, reverse_mode=True)
            point_records.append(point_record)
            summaries.append(
                {
                    "file": str(csv_path),
                    "solver_status": result.solver_status,
                    "termination": result.solver_termination,
                    "dli_target_found": result.dli_target_found,
                    "dli_avg": result.dli_avg,
                    "drr_percent": result.drr,
                    "timeseries_csv": timeseries_path,
                }
            )
        else:
            # Forward mode: solve directly
            print(f"  Running optimization solver (DLI={config.control.dli_target})...")
            try:
                result = solve_for_weather(csv_path, metadata, weather_data, config, args.solver, args.solver_executable)
                print(f"  Complete! PRR={result.prr:.2f}%")
                timeseries_path, point_record = write_outputs(
                    result, output_dir, reverse_mode=False, dli_target_used=config.control.dli_target
                )
                point_records.append(point_record)
                summaries.append(
                    {
                        "file": str(csv_path),
                        "solver_status": result.solver_status,
                        "termination": result.solver_termination,
                        "prr_percent": result.prr,
                        "dli_avg": point_record.get("dli_avg", 0.0),
                        "drr_percent": point_record.get("drr_percent", 0.0),
                        "timeseries_csv": timeseries_path,
                    }
                )
            except RuntimeError as e:
                print(f"  Skipped! Solver failed: {e}")
                # Record failed point
                summaries.append(
                    {
                        "file": str(csv_path),
                        "solver_status": "failed",
                        "termination": "infeasible",
                        "prr_percent": None,
                        "dli_avg": None,
                        "drr_percent": None,
                        "timeseries_csv": "",
                    }
                )

    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / "summary.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    if point_records:
        point_df = pd.DataFrame(point_records)
        # Select output filename based on mode
        if reverse_mode:
            point_path = output_dir / "points_dli.csv"
        else:
            point_path = output_dir / "points_prr.csv"
        point_df.to_csv(point_path, index=False)
    else:
        point_path = None

    print("\nResults Summary:")
    print(summary_df.to_string(index=False))
    print(f"\nSummary file saved: {summary_path}")
    if point_path:
        print(f"GIS point file: {point_path}")


if __name__ == "__main__":
    main()
