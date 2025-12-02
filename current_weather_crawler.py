import requests
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional

def load_amedas_stations(filepath: str = "amedastable.json") -> Dict[str, str]:
    """
    Loads the AMeDAS station list from a JSON file.
    Maps the Japanese station name (kjName) to its station code.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create a dictionary mapping the Japanese name to the station code
        return {details["kjName"]: code for code, details in data.items()}
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not load AMeDAS station file '{filepath}'. {e}")
        return {}


def _check_quality(data: list) -> Optional[float]:
    """
    Checks the data quality flag from the AMeDAS data.
    Returns the value if quality is good (flag == 0), otherwise None.
    """
    if isinstance(data, list) and len(data) == 2 and data[1] == 0:
        return data[0]
    return None


def get_current_conditions(area_name: str, station_map: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """
    Fetches and parses the latest observed weather conditions for a specified area.

    Args:
        area_name: The name of the area (e.g., "Tokyo").
                   Must be a key in the AMEDAS_STATIONS dictionary.
    Returns:
        A dictionary containing the parsed observation data, or None on failure.
    """
    # 1. Validate the input area name
    station_code = station_map.get(area_name)
    if not station_code:
        print(f"Error: AMeDAS station for '{area_name}' not found.")
        print("Please choose from the following available stations:")
        # Print first 10 as an example
        for i, station in enumerate(station_map.keys()):
            if i >= 10: break
            print(f"- {station}")
        return

    # 2. Construct the API URL using the new 'point' data source.
    # Add a browser-like User-Agent header to avoid being blocked.
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        # This endpoint requires a timestamp rounded down to the nearest 3-hour block.
        # Get the current time in Japan's timezone (JST, UTC+9).
        jst = timezone(timedelta(hours=9))
        now_jst = datetime.now(jst)

        # Calculate the correct 3-hour block for the URL.
        date_str = now_jst.strftime('%Y%m%d')
        hour_block_str = f"{(now_jst.hour // 3) * 3:02d}" # Formats as 00, 03, 06...

        data_url = f"https://www.jma.go.jp/bosai/amedas/data/point/{station_code}/{date_str}_{hour_block_str}.json"
        print(f"Fetching current conditions for {area_name} ({station_code}) from {data_url} ...\n")

        # 3. Fetch and parse the observation data
        data_response = requests.get(data_url, headers=headers)
        data_response.raise_for_status()
        station_log = data_response.json()

        # 4. Find the most recent observation in the returned 24-hour log.
        # The keys are timestamps like "20231027140000". We want the latest one.
        latest_timestamp = max(station_log.keys())
        station_data = station_log[latest_timestamp]

        if not station_data:
            print(f"Could not find observation data for station {station_code} in the response.")
            return None

        # 5. Extract and structure the information
        obs_datetime_obj = datetime.strptime(latest_timestamp, '%Y%m%d%H%M%S')
        temp = _check_quality(station_data.get("temp"))
        humidity = _check_quality(station_data.get("humidity"))
        wind_speed = _check_quality(station_data.get("wind"))
        wind_dir_code = _check_quality(station_data.get("windDirection"))
        precip = _check_quality(station_data.get("precipitation1h"))

        return {
            "stationName": area_name,
            "observationTime": obs_datetime_obj.replace(tzinfo=jst).isoformat(),
            "temperature": temp,
            "humidity": humidity,
            "windSpeed": wind_speed,
            "windDirectionCode": wind_dir_code,
            "precipitation1h": precip
        }

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching data: {e}")
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error processing observation data: {e}")
    return None

def print_current_conditions(data: Dict[str, Any]):
    """Prints a formatted summary of the current conditions."""
    obs_datetime = datetime.fromisoformat(data['observationTime']).strftime('%Y-%m-%d %H:%M:%S')
    wind_directions = ["北","北北東","北東","東北東","東","東南東","南東","南南東","南","南南西","南西","西南西","西","西北西","北西","北北西","静穏"]
    wind_dir_code = data['windDirectionCode']
    wind_direction = wind_directions[wind_dir_code] if isinstance(wind_dir_code, int) and wind_dir_code <= 16 else "N/A"

    print(f"--- 現在の気象状況 (Current Weather Conditions) ---")
    print(f"観測地点 (Station): {data['stationName']}")
    print(f"観測日時 (Observation Time): {obs_datetime}")
    print("-" * 40)
    print(f"  気温 (Temperature): {data['temperature']}°C\n  湿度 (Humidity): {data['humidity']}%\n  風速 (Wind Speed): {data['windSpeed']} m/s\n  風向 (Wind Direction): {wind_direction}\n  1時間降水量 (1h Precipitation): {data['precipitation1h']} mm")

def save_current_conditions(area_name: str, data: Dict[str, Any]):
    """Saves the current conditions data to a timestamped JSON file."""
    reports_dir = "current_weather_reports"
    os.makedirs(reports_dir, exist_ok=True)

    obs_dt = datetime.fromisoformat(data['observationTime'])
    timestamp = obs_dt.strftime('%Y%m%d_%H%M%S')
    filename = f"{area_name.lower()}_{timestamp}.json"
    filepath = os.path.join(reports_dir, filename)

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Successfully saved current conditions to {filepath}")
    except IOError as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    # Load the full list of AMeDAS stations from the JSON file.
    amedas_station_map = load_amedas_stations("amedastable.json")

    if not amedas_station_map:
        print("Exiting: AMeDAS station map could not be loaded.")
    else:
        # Use Japanese names to look up stations
        cities_to_check = ["東京", "大阪"]

        for city in cities_to_check:
            print(f"--- Processing Current Conditions for {city} ---")
            current_data = get_current_conditions(city, amedas_station_map)

            if current_data:
                print_current_conditions(current_data)
                save_current_conditions(city, current_data)
            
            print("\n" + "="*60 + "\n")
