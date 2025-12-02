import requests
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

def load_forecast_areas(filepath: str = "area.json") -> Dict[str, str]:
    """
    Loads the forecast area list from area.json.
    Maps the Japanese office name (e.g., "東京都") to its area code.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # The 'offices' section contains the top-level prefectural codes we need.
        return {details["name"]: code for code, details in data.get("offices", {}).items()}
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not load forecast area file '{filepath}'. {e}")
        return {}

def get_weather_forecast(area_name: str, area_map: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """
    Fetches and parses the weather forecast for a specified Japanese area.

    Args:
        area_name: The Japanese name of the area (e.g., "東京都").
        area_map: A dictionary mapping area names to codes.
    Returns:
        A dictionary containing the parsed forecast data, or None on failure.
    """
    # 1. Validate the input area name
    area_code = area_map.get(area_name)
    if not area_code:
        print(f"Error: Area '{area_name}' not found.")
        print("Please choose from the following available areas:")
        # Print first 10 as an example
        for i, station in enumerate(area_map.keys()):
            if i >= 10: break
            print(f"- {station}")
        return None

    # 2. Construct the API URL
    url = f"https://www.jma.go.jp/bosai/forecast/data/forecast/{area_code}.json"
    print(f"Fetching weather data for {area_name} ({area_code}) from {url} ...\n")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        # 3. Fetch the data from the JMA server
        response = requests.get(url, headers=headers)
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()
        
        # 4. Parse the JSON response
        # The response is a list, we'll use the first element which contains the main forecast.
        data = response.json()
        if not data:
            print("Error: No weather data returned from the server.")
            return None
            
        forecast_data = data[0]

        # 5. Extract and display the information
        publishing_office = forecast_data.get("publishingOffice", "N/A")
        report_datetime = forecast_data.get("reportDatetime", "N/A")
        
        # The forecast is divided into time series. We need to find the correct series
        # for weather, precipitation (pops), and temperature (temps).
        time_series = forecast_data.get("timeSeries", [])
        if not time_series:
            print("No time series data available in the forecast.")
            return None

        # Find the relevant time series by checking for key fields
        weather_info = next((ts for ts in time_series if "weathers" in ts["areas"][0]), None)
        pops_info = next((ts for ts in time_series if "pops" in ts["areas"][0]), None)
        temp_info = next((ts for ts in time_series if "temps" in ts["areas"][0]), None)

        if not weather_info or not temp_info:
            print("Error: Essential weather or temperature data is missing from the response.")
            return None

        # --- Robust Data Extraction ---
        # 1. Get the main weather area's information. We assume the first area is the primary one.
        main_weather_area = weather_info["areas"][0]
        main_area_name = main_weather_area.get('area', {}).get('name', 'N/A')

        # 2. Initialize forecast structure based on the weather's time definitions.
        daily_forecasts = {}
        for i, date_str in enumerate(weather_info.get("timeDefines", [])):
            # We only care about the first two days (today and tomorrow)
            if i >= 2:
                break
            date = date_str[:10]
            daily_forecasts[date] = {
                "weather": main_weather_area.get("weathers", ["N/A"])[i],
                "wind": main_weather_area.get("winds", ["N/A"])[i],
                "wave": main_weather_area.get("waves", ["N/A"])[i],
                "min_temp": "N/A",
                "max_temp": "N/A",
                "pops": []
            }

        # 3. Extract temperature data.
        # The temp area (e.g., "東京") is different from the weather area ("東京地方").
        # We assume the first temp area in the list corresponds to the first weather area.
        main_temp_area = temp_info["areas"][0]
        temps = main_temp_area.get("temps", [])
        temp_times = temp_info.get("timeDefines", [])

        # Create a mapping of date to its min/max temps.
        # This robustly handles the non-chronological order of timeDefines.
        temp_map = {}
        for time_str, temp_val in zip(temp_times, temps):
            date = time_str[:10]
            if date not in temp_map:
                temp_map[date] = {"min": None, "max": None}
            
            # In JMA data, 00:00 is typically min, 09:00 is max for the day.
            if "T00:00:00" in time_str:
                temp_map[date]["min"] = temp_val
            elif "T09:00:00" in time_str:
                temp_map[date]["max"] = temp_val

        for date, temps in temp_map.items():
            if date in daily_forecasts:
                daily_forecasts[date]["min_temp"] = temps.get("min", "N/A")
                daily_forecasts[date]["max_temp"] = temps.get("max", "N/A")

        # 4. Extract precipitation probability (pops).
        if pops_info:
            # We assume the first pops area corresponds to the first weather area.
            main_pops_area = pops_info["areas"][0]
            pops_values = main_pops_area.get("pops", [])
            pops_times = pops_info.get("timeDefines", [])
            
            for i, time_str in enumerate(pops_times):
                date = time_str[:10]
                if date in daily_forecasts and i < len(pops_values):
                    hour = int(time_str[11:13])
                    # Determine the 6-hour block
                    if 0 <= hour < 6: slot = "00-06時"
                    elif 6 <= hour < 12: slot = "06-12時"
                    elif 12 <= hour < 18: slot = "12-18時"
                    else: slot = "18-24時"
                    
                    daily_forecasts[date]["pops"].append(f"{slot}: {pops_values[i]}%")

        return {
            "publishingOffice": publishing_office,
            "reportDatetime": report_datetime,
            "areaName": main_area_name,
            "forecasts": daily_forecasts
        }

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching data: {e}")
    except json.JSONDecodeError:
        print("Error: Failed to parse the server's response. It might not be valid JSON.")
    except (IndexError, KeyError) as e:
        print(f"Error: Could not find expected data in the response. The data structure may have changed. Details: {e}")
    return None

def print_forecast(data: Dict[str, Any]):
    """Prints a formatted forecast from the data dictionary."""
    print(f"発表機関 (Publishing Office): {data['publishingOffice']}")
    print(f"発表日時 (Report Datetime): {data['reportDatetime']}")
    print("-" * 30)
    for i, (date, forecast) in enumerate(data['forecasts'].items()):
        day_label = "今日" if i == 0 else "明日"
        print(f"--- {day_label} ({date}) の天気 ---")
        print(f"  地域 (Area): {data['areaName']}")
        print(f"  天気 (Weather): {forecast['weather']}")
        print(f"  風 (Wind): {forecast['wind']}")
        if forecast['wave'] != "N/A":
            print(f"  波 (Wave): {forecast['wave']}")
        min_t, max_t = forecast['min_temp'], forecast['max_temp']
        print(f"  気温 (Temperature): 最低 {min_t}°C / 最高 {max_t}°C")
        if forecast['pops']:
            print(f"  降水確率 (Precipitation): {', '.join(forecast['pops'])}")
        print()

def save_forecast(area_name: str, data: Dict[str, Any]):
    """Saves the forecast data to a timestamped JSON file."""
    reports_dir = "forecast_weather_reports"
    os.makedirs(reports_dir, exist_ok=True)

    # Create a clean timestamp for the filename
    report_dt = datetime.fromisoformat(data['reportDatetime'])
    timestamp = report_dt.strftime('%Y%m%d_%H%M%S')
    filename = f"{area_name.lower()}_{timestamp}.json"
    filepath = os.path.join(reports_dir, filename)

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Successfully saved forecast to {filepath}")
    except IOError as e:
        print(f"Error saving file: {e}")


# --- Example Usage ---
if __name__ == "__main__":
    # Load the full list of forecast areas from the JSON file.
    forecast_area_map = load_forecast_areas("area.json")

    if not forecast_area_map:
        print("Exiting: Forecast area map could not be loaded.")
    else:
        # Use Japanese names to look up areas
        areas_to_check = ["東京都", "大阪府"]

        for area in areas_to_check:
            print(f"--- Processing Forecast for {area} ---")
            forecast_data = get_weather_forecast(area, forecast_area_map)

            if forecast_data:
                print_forecast(forecast_data)
                save_forecast(area, forecast_data)
            
            print("\n" + "="*60 + "\n")
