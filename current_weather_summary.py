import os
import json
from datetime import datetime
from collections import defaultdict

def summarize_current_reports(directory: str = "current_weather_reports"):
    """
    Reads all JSON observation files in a directory and summarizes the changes for each city.
    """
    print(f"--- Analyzing Current Condition Reports in '{directory}' ---\n")

    try:
        files = [f for f in os.listdir(directory) if f.endswith('.json')]
        if not files:
            print(f"No report files found in '{directory}'.")
            print("Run current_weather_crawler.py first to generate reports.")
            return
        
        files.sort()

        # Group reports by city
        city_reports = defaultdict(list)
        for filename in files:
            city_name = filename.split('_')[0]
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    city_reports[city_name].append(json.load(f))
            except (json.JSONDecodeError, KeyError):
                print(f"Warning: Skipping corrupted or invalid file: {filename}")

        # Display summary for each city
        wind_directions = ["北","北北東","北東","東北東","東","東南東","南東","南南東","南","南南西","南西","西南西","西","西北西","北西","北北西","静穏"]

        for city, reports in city_reports.items():
            print(f"--- Observed Conditions History for {city.capitalize()} ---")
            print(f"{'Observation Time':<20} | {'Temp (°C)':<10} | {'Humidity (%)':<12} | {'Wind (m/s)':<15} | {'Precip (mm/h)':<15}")
            print("-" * 85)

            for report in reports:
                obs_time = datetime.fromisoformat(report['observationTime']).strftime('%Y-%m-%d %H:%M')
                temp = report.get('temperature', 'N/A')
                humidity = report.get('humidity', 'N/A')
                
                wind_speed = report.get('windSpeed', 'N/A')
                wind_dir_code = report.get('windDirectionCode')
                wind_dir = wind_directions[wind_dir_code] if isinstance(wind_dir_code, int) and wind_dir_code <= 16 else ""
                wind_str = f"{wind_speed} {wind_dir}".strip()

                precip = report.get('precipitation1h', 'N/A')

                print(f"{obs_time:<20} | {str(temp):<10} | {str(humidity):<12} | {wind_str:<15} | {str(precip):<15}")
            print("\n")

    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
        print("Please run current_weather_crawler.py first to create it.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    summarize_current_reports()