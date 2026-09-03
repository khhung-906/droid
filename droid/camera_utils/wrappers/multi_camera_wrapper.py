import logging
import os
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from droid.camera_utils.camera_readers.zed_camera import gather_zed_cameras
from droid.camera_utils.info import get_camera_type


class MultiCameraWrapper:
    def __init__(self, camera_kwargs={}):
        # Background reading state. read_cameras() defaults to serving frames from a
        # daemon grab thread (see read_cameras) so callers never block on grab(). This
        # must be initialized before any camera setup because set_trajectory_mode()
        # touches it.
        self._bg_thread = None
        self._bg_stop = None
        self._bg_lock = threading.Lock()
        self._bg_latest = None
        self._recording = False

        # Open Cameras #
        zed_cameras = gather_zed_cameras()
        self.camera_dict = {cam.serial_number: cam for cam in zed_cameras}

        # Set Correct Parameters #
        for cam_id in self.camera_dict.keys():
            cam_type = get_camera_type(cam_id)
            curr_cam_kwargs = camera_kwargs.get(cam_type, {})
            self.camera_dict[cam_id].set_reading_parameters(**curr_cam_kwargs)

        # Launch Camera #
        self.set_trajectory_mode()

    ### Calibration Functions ###
    def get_camera(self, camera_id):
        return self.camera_dict[camera_id]

    def enable_advanced_calibration(self):
        for cam in self.camera_dict.values():
            cam.enable_advanced_calibration()

    def disable_advanced_calibration(self):
        for cam in self.camera_dict.values():
            cam.disable_advanced_calibration()

    def set_calibration_mode(self, cam_id):
        # Stop the background grab thread before reconfiguring cameras.
        self.stop_background_reading()
        # If High Res Calibration, Only One Can Run #
        close_all = any([cam.high_res_calibration for cam in self.camera_dict.values()])

        if close_all:
            for curr_cam_id in self.camera_dict:
                if curr_cam_id != cam_id:
                    self.camera_dict[curr_cam_id].disable_camera()

        self.camera_dict[cam_id].set_calibration_mode()

    def set_trajectory_mode(self):
        # Stop the background grab thread before reconfiguring cameras; read_cameras()
        # restarts it lazily once cameras are back in trajectory mode.
        self.stop_background_reading()
        # If High Res Calibration, Close All #
        close_all = any(
            [cam.high_res_calibration and cam.current_mode == "calibration" for cam in self.camera_dict.values()]
        )

        if close_all:
            for cam in self.camera_dict.values():
                cam.disable_camera()

        # Put All Cameras In Trajectory Mode #
        for cam in self.camera_dict.values():
            cam.set_trajectory_mode()

    ### Data Storing Functions ###
    def start_recording(self, recording_folderpath):
        # SVO recording writes one frame per grab(). Take grab() back onto the caller's
        # thread (one frame per read_cameras call) instead of the free-running background
        # thread, so the SVO cadence matches the control loop as before.
        self.stop_background_reading()
        self._recording = True
        subdir = os.path.join(recording_folderpath, "SVO")
        if not os.path.isdir(subdir):
            os.makedirs(subdir)
        for cam in self.camera_dict.values():
            filepath = os.path.join(subdir, cam.serial_number + ".svo")
            cam.start_recording(filepath)

    def stop_recording(self):
        for cam in self.camera_dict.values():
            cam.stop_recording()
        # read_cameras() resumes background grabbing on the next call.
        self._recording = False

    ### Basic Camera Functions ###
    def read_cameras(self):
        # Default path: a daemon thread owns grab() and we return the latest cached frame
        # without blocking on grab(). Fall back to a synchronous per-call grab when SVO
        # recording is active (preserve one-frame-per-read SVO cadence) or during
        # calibration (which needs the frame at the exact current pose, not a stale one).
        if self._recording or self._in_calibration_mode():
            return self._read_cameras_blocking()
        if self._bg_thread is None or not self._bg_thread.is_alive():
            self.start_background_reading()
        latest = self._wait_for_background_frame()
        if latest is not None:
            return latest
        logging.getLogger(__name__).warning(
            "Background camera reader produced no frame in time; doing a direct read."
        )
        return self._read_cameras_blocking()

    def _in_calibration_mode(self):
        return any(
            getattr(cam, "current_mode", None) == "calibration"
            for cam in self.camera_dict.values()
        )

    def _read_cameras_blocking(self):
        full_obs_dict = defaultdict(dict)
        full_timestamp_dict = {}

        running_cam_ids = [
            cam_id for cam_id in self.camera_dict
            if self.camera_dict[cam_id].is_running()
        ]

        if len(running_cam_ids) <= 1:
            for cam_id in running_cam_ids:
                data_dict, timestamp_dict = self.camera_dict[cam_id].read_camera()
                for key in data_dict:
                    full_obs_dict[key].update(data_dict[key])
                full_timestamp_dict.update(timestamp_dict)
        else:
            def _read(cam_id):
                return cam_id, self.camera_dict[cam_id].read_camera()

            with ThreadPoolExecutor(max_workers=len(running_cam_ids)) as executor:
                futures = {executor.submit(_read, cam_id): cam_id for cam_id in running_cam_ids}
                for future in as_completed(futures):
                    cam_id, (data_dict, timestamp_dict) = future.result()
                    for key in data_dict:
                        full_obs_dict[key].update(data_dict[key])
                    full_timestamp_dict.update(timestamp_dict)

        return full_obs_dict, full_timestamp_dict

    def _wait_for_background_frame(self, timeout=10.0):
        """Return the most recent cached (obs, timestamp) tuple, waiting up to `timeout`
        seconds for the first frame. Returns None only if it times out."""
        deadline = time.time() + timeout
        while True:
            with self._bg_lock:
                latest = self._bg_latest
            if latest is not None:
                return latest
            if time.time() >= deadline:
                return None
            time.sleep(0.002)

    def start_background_reading(self):
        """Spawn a daemon thread that continuously grabs frames into a cache so that
        read_cameras() never blocks on grab(). Idempotent; the latest frame may be up
        to ~one grab-period stale."""
        if self._bg_thread is not None and self._bg_thread.is_alive():
            return
        self._bg_stop = threading.Event()
        # Cap the grab rate to ~the control rate. Measured: read_cameras() returns in
        # ~14ms (grab() does NOT block to the 30fps frame clock -- it returns buffered
        # frames), so the unthrottled loop spins at ~71fps, and ~60% of those grabs are
        # redundant (the camera only refreshes at 30fps). Each redundant iteration still
        # does 3x retrieve + cvtColor, holding the GIL and contending with the control
        # loop. Throttling to 30 Hz (the camera's true update rate) removes that waste
        # and roughly halves the reader's GIL/CPU load. Tunable via bg_target_period.
        target_period = getattr(self, "bg_target_period", 1.0 / 30.0)

        def _loop():
            while not self._bg_stop.is_set():
                t0 = time.time()
                try:
                    result = self._read_cameras_blocking()
                except Exception:
                    # Transient grab failure: skip and retry on the next iteration.
                    continue
                with self._bg_lock:
                    self._bg_latest = result
                if target_period:
                    remaining = target_period - (time.time() - t0)
                    if remaining > 0:
                        # Interruptible sleep so stop() shuts down promptly.
                        self._bg_stop.wait(remaining)

        self._bg_thread = threading.Thread(target=_loop, name="camera-bg-reader", daemon=True)
        self._bg_thread.start()

    def stop_background_reading(self):
        """Stop the background reader thread (if running)."""
        if self._bg_stop is not None:
            self._bg_stop.set()
        if self._bg_thread is not None:
            self._bg_thread.join(timeout=2.0)
        self._bg_thread = None
        self._bg_latest = None

    def disable_cameras(self):
        self.stop_background_reading()
        for camera in self.camera_dict.values():
            camera.disable_camera()
