import numpy as np
import math

class GeometryTransformer:
    """
    Handles coordinate transformations for AR projection.
    Follows the pipeline: WGS84 -> LTP -> Body -> Velodyne -> Camera -> Image
    """

    def __init__(self, calib_data):
        """
        Initialize with calibration data.
        
        Args:
            calib_data (dict): Dictionary containing calibration matrices:
                - 'Tr_velo_to_cam': 4x4 list/array
                - 'R_rect': 3x3 or 4x4 list/array
                - 'P2': 3x4 list/array
        """
        # Load calibration matrices
        self.Tr_velo_to_cam = np.array(calib_data['Tr_velo_to_cam'])
        
        # R_rect might be 3x3, need to expand to 4x4 for homogeneous ops
        R_rect_in = np.array(calib_data['R_rect'])
        if R_rect_in.shape == (3, 3):
            self.R_rect = np.eye(4)
            self.R_rect[:3, :3] = R_rect_in
        elif R_rect_in.shape == (4, 4):
            self.R_rect = R_rect_in
        else:
            raise ValueError(f"Invalid R_rect shape: {R_rect_in.shape}")

        self.P2 = np.array(calib_data['P2'])
        
        # IMU to Velo is usually fixed/identity-like in KITTI but strictly:
        # We need Tr_imu_to_velo. If not provided, we might assume identity or specific values.
        # The report mentions specific values. For now, we'll assume it's passed or use default.
        # If not in calib_data, we use the values from the report/calib file.
        if 'Tr_imu_to_velo' in calib_data:
            self.Tr_imu_to_velo = np.array(calib_data['Tr_imu_to_velo'])
        else:
            # Default from 2011_10_03 calib_imu_to_velo.txt (approximate)
            # R is approx identity, t is approx [-0.81, 0.32, -0.80]
            self.Tr_imu_to_velo = np.array([
                [0.9999976, 0.0007553, -0.0020358, -0.8086759],
                [-0.0007854, 0.9998898, -0.0148229, 0.3195559],
                [0.0020244, 0.0148245, 0.9998881, -0.7997231],
                [0.0, 0.0, 0.0, 1.0]
            ])

    def wgs84_to_ltp(self, lat, lon, alt, ref_lat, ref_lon, ref_alt):
        """
        Convert WGS84 (lat, lon, alt) to Local Tangent Plane (x, y, z).
        Using Flat Earth approximation for short distances.
        
        Args:
            lat, lon, alt: Target point
            ref_lat, ref_lon, ref_alt: Reference point (Vehicle current position)
            
        Returns:
            np.array: [x, y, z] in meters (East, North, Up)
        """
        R_e = 6378137.0  # Earth radius in meters
        
        d_lat = np.radians(lat - ref_lat)
        d_lon = np.radians(lon - ref_lon)
        
        # x is East, y is North
        x = d_lon * R_e * np.cos(np.radians(ref_lat))
        y = d_lat * R_e
        z = alt - ref_alt
        
        return np.array([x, y, z])

    def ltp_to_body(self, point_ltp, yaw, pitch, roll):
        """
        Convert point from LTP (East, North, Up) to Body Frame (Forward, Left, Up).
        
        KITTI Yaw definition: 0 = East, Positive = Counter-Clockwise.
        Standard Rotation Matrices usually rotate a vector IN a frame.
        Here we want to transform the COORDINATES of a fixed point from World to Body.
        P_body = R_body_to_world^T * P_world
        
        R_body_to_world = R_z(yaw) * R_y(pitch) * R_x(roll)
        
        Args:
            point_ltp: [x, y, z] (East, North, Up)
            yaw, pitch, roll: Vehicle attitude in radians (KITTI definition)
            
        Returns:
            np.array: [x, y, z] (Forward, Left, Up)
        """
        # Rotation matrices
        c_y, s_y = np.cos(yaw), np.sin(yaw)
        c_p, s_p = np.cos(pitch), np.sin(pitch)
        c_r, s_r = np.cos(roll), np.sin(roll)
        
        # R_z (Yaw) - Note: KITTI Yaw 0 is East, which matches LTP X axis.
        # So standard R_z works for East-North plane.
        R_z = np.array([
            [c_y, -s_y, 0],
            [s_y, c_y, 0],
            [0, 0, 1]
        ])
        
        # R_y (Pitch)
        R_y = np.array([
            [c_p, 0, s_p],
            [0, 1, 0],
            [-s_p, 0, c_p]
        ])
        
        # R_x (Roll)
        R_x = np.array([
            [1, 0, 0],
            [0, c_r, -s_r],
            [0, s_r, c_r]
        ])
        
        # Combined Rotation (Body to World)
        R = R_z @ R_y @ R_x
        
        # World to Body is inverse (transpose)
        R_inv = R.T
        
        return R_inv @ point_ltp

    def body_to_image(self, point_body):
        """
        Convert point from Body Frame to Image Pixel Coordinates.
        Chain: Body -> IMU -> Velo -> Cam_Unrect -> Cam_Rect -> Image
        
        Args:
            point_body: [x, y, z]
            
        Returns:
            tuple: (u, v, w) where w is depth. Returns None if behind camera.
        """
        # Homogeneous coordinates
        p_body_h = np.append(point_body, 1.0)
        
        # 1. Body (IMU) -> Velodyne
        # Note: Usually Body origin is IMU.
        p_velo = self.Tr_imu_to_velo @ p_body_h
        
        # 2. Velodyne -> Camera (Unrectified)
        p_cam_unrect = self.Tr_velo_to_cam @ p_velo
        
        # 3. Rectification
        p_cam_rect = self.R_rect @ p_cam_unrect
        
        # 4. Projection to Image Plane
        # P2 is 3x4, p_cam_rect is 4x1
        p_img_h = self.P2 @ p_cam_rect
        
        # Normalize
        w = p_img_h[2]
        if w <= 0:
            return None # Behind camera
            
        u = p_img_h[0] / w
        v = p_img_h[1] / w
        
        return (u, v, w)

    def transform_path(self, path_gps, current_gps_state):
        """
        Transform a list of GPS points to Image coordinates.
        
        Args:
            path_gps: List of dicts/tuples {'lat':, 'lon':, 'alt':}
            current_gps_state: dict with keys 'lat', 'lon', 'alt', 'yaw', 'pitch', 'roll'
            
        Returns:
            list: List of (u, v) tuples valid for rendering.
        """
        ref_lat = current_gps_state['lat']
        ref_lon = current_gps_state['lon']
        ref_alt = current_gps_state['alt']
        
        yaw = current_gps_state['yaw']
        pitch = current_gps_state['pitch']
        roll = current_gps_state['roll']
        
        projected_points = []
        
        for point in path_gps:
            # 1. WGS84 -> LTP
            p_ltp = self.wgs84_to_ltp(
                point['lat'], point['lon'], point.get('alt', ref_alt),
                ref_lat, ref_lon, ref_alt
            )
            
            # 2. LTP -> Body
            p_body = self.ltp_to_body(p_ltp, yaw, pitch, roll)
            
            # 3. Body -> Image
            res = self.body_to_image(p_body)
            
            if res:
                u, v, w = res
                projected_points.append((u, v))
                
        return projected_points
