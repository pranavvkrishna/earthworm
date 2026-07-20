# Weather Tool
# Geocodes a location name to lat/lon using Open-Meteo's free geocoding
# API, then fetches a forecast from the National Weather Service (NWS) API.

import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
NWS_POINTS_URL = "https://api.weather.gov/points/{lat},{lon}"

# NWS requires a descriptive User-Agent identifying the app/contact
HEADERS = {
    "User-Agent": "EarthwormAI (student project, contact: example@example.com)"
}


def geocode_location(location: str):
    """Turn a location name (city, zip, address) into lat/lon."""
    # Open-Meteo's geocoder works best with just a city name — strip
    # anything after a comma (e.g. "Sacramento, CA" -> "Sacramento")
    clean_location = location.split(",")[0].strip()

    params = {"name": clean_location, "count": 1, "format": "json"}
    response = requests.get(GEOCODE_URL, params=params, timeout=10)
    response.raise_for_status()
    results = response.json()

    if not results.get("results"):
        return None

    top = results["results"][0]
    return top["latitude"], top["longitude"]


def get_forecast(lat: float, lon: float):
    # fetch a short-term forecast from NWS for given coordinates
    # step 1: get the forecast endpoint for this location
    points_resp = requests.get(NWS_POINTS_URL.format(lat=lat, lon=lon), headers=HEADERS, timeout=10)
    points_resp.raise_for_status()
    forecast_url = points_resp.json()["properties"]["forecast"]

    # step 2: fetch the actual forecast
    forecast_resp = requests.get(forecast_url, headers=HEADERS, timeout=10)
    forecast_resp.raise_for_status()
    periods = forecast_resp.json()["properties"]["periods"]

    return periods


def get_weather_summary(location: str, num_periods: int = 3):
    # full pipeline: location name -> forecast summary string
    coords = geocode_location(location)
    if coords is None:
        return f"Could not find a location matching '{location}'."

    lat, lon = coords

    try:
        periods = get_forecast(lat, lon)
    except requests.exceptions.RequestException as e:
        return f"Could not retrieve forecast for {location}: {str(e)}"

    if not periods:
        return f"No forecast data available for {location}."

    lines = [f"Weather forecast for {location}:"]
    for period in periods[:num_periods]:
        lines.append(
            f"- {period['name']}: {period['temperature']}°{period['temperatureUnit']}, "
            f"{period['shortForecast']}"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    test_location = "Sacramento, CA"
    print(get_weather_summary(test_location))