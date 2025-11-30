import os
import requests
import numpy as np
import math
from pathlib import Path

class GoogleMapsClient:
    """
    Client for Google Maps Roads API.
    Handles fetching snapped road geometry for AR projection.
    """
    
    BASE_URL = "https://roads.googleapis.com/v1/snapToRoads"
    
    def __init__(self, api_key=None):
        # Try loading from .env file in project root
        try:
            env_path = Path(__file__).resolve().parent.parent / '.env'
            if env_path.exists():
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            if key.strip() == 'GOOGLE_MAPS_API_KEY':
                                os.environ[key.strip()] = value.strip().strip('"\'')
                                print(f"[MapsClient] Loaded API Key from {env_path}")
        except Exception as e:
            print(f"[MapsClient] Failed to load .env: {e}")

        self.api_key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            print("[MapsClient] Warning: No API Key provided. Using MOCK mode.")
            
    def get_snapped_path(self, current_gps, yaw, lookahead_dist=50.0, num_samples=10):
        """
        Get a smooth path snapped to the road.
        
        Args:
            current_gps (dict): {'lat': float, 'lon': float}
            yaw (float): Current vehicle yaw (KITTI definition: 0=East, CCW+)
            lookahead_dist (float): How far ahead to query in meters.
            
        Returns:
            list: List of dicts [{'lat':, 'lon':, 'alt':}] representing the road center.
        """
        if not self.api_key:
            return self._get_mock_path(current_gps, yaw, lookahead_dist)
            
        # 1. Calculate lookahead point
        # Convert lat/lon to meters approx
        R_e = 6378137.0
        lat_rad = np.radians(current_gps['lat'])
        
        # Yaw is 0=East, so:
        # dx = dist * cos(yaw)
        # dy = dist * sin(yaw)
        dx = lookahead_dist * np.cos(yaw)
        dy = lookahead_dist * np.sin(yaw)
        
        # Convert back to d_lat, d_lon
        d_lat = dy / R_e
        d_lon = dx / (R_e * np.cos(lat_rad))
        
        target_lat = current_gps['lat'] + np.degrees(d_lat)
        target_lon = current_gps['lon'] + np.degrees(d_lon)
        
        # 2. Construct Path for API
        # We send [current, target] and ask for interpolation
        path_str = f"{current_gps['lat']},{current_gps['lon']}|{target_lat},{target_lon}"
        
        params = {
            'path': path_str,
            'interpolate': 'true',
            'key': self.api_key
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params)
            data = response.json()
            
            if 'snappedPoints' not in data:
                print(f"[MapsClient] Error in API response: {data}")
                return self._get_mock_path(current_gps, yaw, lookahead_dist)
                
            result_path = []
            for item in data['snappedPoints']:
                loc = item['location']
                result_path.append({
                    'lat': loc['latitude'],
                    'lon': loc['longitude'],
                    'alt': current_gps.get('alt', 0) # API doesn't return alt usually
                })
                
            return result_path
            
        except Exception as e:
            print(f"[MapsClient] Request failed: {e}")
            return self._get_mock_path(current_gps, yaw, lookahead_dist)

    def _get_mock_path(self, current_gps, yaw, dist):
        """
        Generate a straight line path for testing without API.
        """
        path = []
        R_e = 6378137.0
        lat_rad = np.radians(current_gps['lat'])
        
        for d in np.linspace(0, dist, 10):
            dx = d * np.cos(yaw)
            dy = d * np.sin(yaw)
            
            d_lat = dy / R_e
            d_lon = dx / (R_e * np.cos(lat_rad))
            
            path.append({
                'lat': current_gps['lat'] + np.degrees(d_lat),
                'lon': current_gps['lon'] + np.degrees(d_lon),
                'alt': current_gps.get('alt', 0)
            })
            
        return path
