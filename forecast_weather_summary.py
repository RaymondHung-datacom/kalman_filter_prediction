import os
import json
from datetime import datetime

def summarize_daily_reports(directory: str = "forecast_weather_reports"):
    """
    Reads all JSON forecast files in a directory and summarizes the changes.
    """
    print(f"--- Analyzing Forecast Reports in '{directory}' ---\n")

    try:
        # 1. Get all report files and sort them chronologically
        files = [f for f in os.listdir(directory) if f.endswith('.json')]
        if not files:
            print(f"No report files found in '{directory}'.")
            print("Run weather_crawler.py first to generate reports.")
            return
        
        files.sort() # Sorts by filename, which includes the timestamp

        # 2. Load all reports into a list
        reports = []
        for filename in files:
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                reports.append(json.load(f))

        # 3. Analyze and display the summary
        # We will focus on how the forecast for "tomorrow" changes over time.
        if not reports:
            return

        # Get the date for "tomorrow" from the first report
        first_report = reports[0]
        if len(first_report['forecasts']) < 2:
            print("Reports do not contain a forecast for 'tomorrow'. Cannot summarize.")
            return
            
        tomorrow_date = list(first_report['forecasts'].keys())[1]
        
        print(f"--- Summary of Forecast Evolution for {tomorrow_date} ---\n")
        print(f"{'Report Time':<20} | {'Weather Forecast':<30} | {'Temperature (Min/Max)':<25} | {'Possibility of Precipitation (%)'}")
        print("-" * 120)

        for report in reports:
            report_time = datetime.fromisoformat(report['reportDatetime']).strftime('%Y-%m-%d %H:%M')
            
            # Find the forecast for the target "tomorrow" date
            tomorrow_forecast = report['forecasts'].get(tomorrow_date)
            
            if tomorrow_forecast:
                weather = tomorrow_forecast['weather']
                temp = f"{tomorrow_forecast['min_temp']}°C / {tomorrow_forecast['max_temp']}°C"
                pops = ', '.join(tomorrow_forecast['pops']) if tomorrow_forecast['pops'] else "N/A"

                # Print a summary line for this report
                print(f"{report_time:<20} | {weather:<30} | {temp:<25} | {pops}")
            
    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
        print("Please run weather_crawler.py first to create it.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    summarize_daily_reports()

