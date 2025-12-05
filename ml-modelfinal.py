import math
import random
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

# ----------------------------
# CONSTANTS
# ----------------------------
GRAVITATIONAL_CONSTANT = 9.81
MASS = 300  # kg
FRONTAL_AREA = 1.1  # m^2
DENSITY_AIR = 1.225  # kg/m^3
BASE_ROLLING_RESISTANCE = 0.015
COEFF_DRAG = 0.2
BATTERY_CAPACITY = 4960  # Wh
SOLAR_PANEL_EFFICIENCY = 0.18

# Motor efficiencies at various speeds (km/h)
motor_efficiencies = {
    10: 0.44, 20: 0.64, 30: 0.74, 40: 0.81, 50: 0.85, 60: 0.90,
}

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def interpolate_efficiency(speed_kph):
    speeds = sorted(motor_efficiencies.keys())
    if speed_kph in motor_efficiencies:
        return motor_efficiencies[speed_kph]
    for i in range(len(speeds) - 1):
        if speeds[i] <= speed_kph < speeds[i + 1]:
            lower, upper = speeds[i], speeds[i + 1]
            return motor_efficiencies[lower] + (motor_efficiencies[upper] - motor_efficiencies[lower]) * (
                (speed_kph - lower) / (upper - lower)
            )
    return motor_efficiencies[speeds[-1]]

def calculate_rolling_resistance_coeff(temperature):
    return BASE_ROLLING_RESISTANCE * (1 + 0.01 * (temperature - 25))

def calculate_remaining_soc(speed_kph, gradient_deg, temperature, wind_speed_mps,
                            ghi10, ghi90, battery_efficiency, current_soc):
    speed_mps = speed_kph * 0.27778
    if speed_mps <= 0:
        return 0
    distance_m = 150_000  # 150 km
    time_s = distance_m / speed_mps

    rolling_resistance = calculate_rolling_resistance_coeff(temperature) * GRAVITATIONAL_CONSTANT * MASS
    drag_force = 0.5 * COEFF_DRAG * FRONTAL_AREA * DENSITY_AIR * (speed_mps + wind_speed_mps) ** 2
    gradient_force = MASS * GRAVITATIONAL_CONSTANT * math.sin(math.radians(gradient_deg))

    resistive_force = rolling_resistance + gradient_force + drag_force
    motor_efficiency = interpolate_efficiency(speed_kph)
    if motor_efficiency == 0:
        return 0

    motor_power_watt = (speed_mps * resistive_force) / (motor_efficiency * 0.97)
    irradiance = (ghi10 + ghi90) / 2
    solar_power = max(0, irradiance * 3.51 * SOLAR_PANEL_EFFICIENCY)
    solar_power *= 1 - 0.004 * max(0, abs(temperature - 25))

    battery_capacity_wh = BATTERY_CAPACITY * battery_efficiency
    energy_used_wh = motor_power_watt * (time_s / 3600)
    energy_generated_wh = solar_power * (time_s / 3600)

    remaining_wh = current_soc * battery_capacity_wh - energy_used_wh + energy_generated_wh
    remaining_soc = (remaining_wh / battery_capacity_wh) * 100
    return max(0, min(remaining_soc, 100))

def find_optimal_speed(gradient_deg, temperature, wind_speed_mps,
                       ghi10, ghi90, battery_efficiency, current_soc):
    """Find speed that maximizes remaining SOC."""
    # Negative because we want to maximize (minimize the negative)
    def objective(speed_kph):
        return -calculate_remaining_soc(speed_kph, gradient_deg, temperature, 
                                       wind_speed_mps, ghi10, ghi90, 
                                       battery_efficiency, current_soc)
    
    # Search for optimal speed between 10-60 km/h
    result = minimize_scalar(objective, bounds=(10, 60), method='bounded')
    optimal_speed = result.x
    max_remaining_soc = -result.fun
    
    return optimal_speed, max_remaining_soc

# ----------------------------
# DATASET GENERATION
# ----------------------------
def generate_dataset(num_points=50000):
    data = []
    for i in range(num_points):
        if i % 5000 == 0:
            print(f"Generating sample {i}/{num_points}...")
        
        # 1. Environmental conditions
        time_of_day = random.uniform(6, 18)  # 6 AM to 6 PM
        ghi10 = max(0, 800 * math.sin((time_of_day - 6) / 12 * math.pi) + random.uniform(-50, 50))
        ghi90 = max(0, 800 * math.sin((time_of_day - 6) / 12 * math.pi) + random.uniform(-50, 50))
        temperature = random.gauss(25 + 0.02 * ((ghi10 + ghi90)/2), 5)
        temperature = max(-5, min(temperature, 45))
        wind_speed = random.uniform(0, 10)
        gradient = random.uniform(-5, 10)

        # 2. Vehicle state
        battery_efficiency = 0.95 - 0.001 * abs(temperature - 25)
        current_soc = random.uniform(0.3, 1.0)

        # 3. Find optimal speed for these specific conditions
        optimal_speed, max_remaining_soc = find_optimal_speed(
            gradient, temperature, wind_speed, ghi10, ghi90, battery_efficiency, current_soc
        )

        # 4. Also calculate at a randomly chosen "actual" speed for comparison
        actual_speed = random.uniform(10, 60)
        actual_remaining_soc = calculate_remaining_soc(actual_speed, gradient, temperature, wind_speed,
                                                       ghi10, ghi90, battery_efficiency, current_soc)

        # 5. Calculate how much SOC is lost by not driving at optimal speed
        soc_loss_from_suboptimal = max_remaining_soc - actual_remaining_soc
        speed_difference = abs(optimal_speed - actual_speed)

        # 6. Synthetic uncertainty
        irradiance_uncertainty = random.gauss(0, 0.1 * ((ghi10 + ghi90)/2))
        temperature_uncertainty = random.gauss(0, 0.05 * temperature)
        remaining_soc_uncertainty = random.gauss(0, 0.05 * actual_remaining_soc)

        data.append([
            time_of_day, ghi10, ghi90, temperature, wind_speed, gradient,
            battery_efficiency, current_soc,
            optimal_speed, max_remaining_soc,
            actual_speed, actual_remaining_soc, 
            soc_loss_from_suboptimal, speed_difference,
            irradiance_uncertainty, temperature_uncertainty, remaining_soc_uncertainty
        ])

    df = pd.DataFrame(data, columns=[
        'time_of_day', 'GHI10', 'GHI90', 'temperature_C', 'wind_mps', 'gradient_deg',
        'battery_efficiency', 'current_soc',
        'optimal_speed_kph', 'optimal_remaining_soc',
        'actual_speed_kph', 'actual_remaining_soc',
        'soc_loss_from_suboptimal', 'speed_difference',
        'irradiance_uncertainty', 'temperature_uncertainty', 'remaining_soc_uncertainty'
    ])
    return df

# ----------------------------
# MAIN EXECUTION
# ----------------------------
if __name__ == "__main__":
    print("Generating dataset with optimal speed calculations...")
    df = generate_dataset(50000)
    df.to_csv("ev_physics_sim_50k_optimal.csv", index=False)
    print("\n✅ Dataset generated successfully! Saved as 'ev_physics_sim_50k_optimal.csv'")
    print("\n📊 Sample data:")
    print(df.head(10))
    print("\n📈 Optimal speed statistics:")
    print(df[['optimal_speed_kph', 'actual_speed_kph', 'soc_loss_from_suboptimal']].describe())