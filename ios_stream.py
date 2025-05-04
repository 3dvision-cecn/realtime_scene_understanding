import numpy as np
import cv2
from threading import Event
from record3d import Record3DStream
import rerun as rr
import signal
import sys
import time


class DemoApp:
    DEVICE_TYPE_TRUEDEPTH = 0
    DEVICE_TYPE_LIDAR = 1

    def __init__(self, dev_idx: int = 0):
        self.event = Event()
        self.session = None
        self.dev_idx = dev_idx

    # -------------------------------------------------------------------- #
    # Callback hooks invoked by Record3DStream
    # -------------------------------------------------------------------- #
    def on_new_frame(self):
        """Wake up the main thread when a fresh RGB-D frame is ready."""
        self.event.set()

    def on_stream_stopped(self):
        print("Stream stopped")

    # -------------------------------------------------------------------- #
    # Helpers
    # -------------------------------------------------------------------- #
    @staticmethod
    def _intrinsic_matrix(coeffs):
        """Record3D -> 3x3 pin-hole camera matrix."""
        return np.array(
            [
                [coeffs.fx, 0.0, coeffs.tx],
                [0.0, coeffs.fy, coeffs.ty],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def _open_device(self):
        devs = Record3DStream.get_connected_devices()
        if len(devs) <= self.dev_idx:
            raise RuntimeError(
                f"No device #{self.dev_idx} - {len(devs)} device(s) found"
            )

        dev = devs[self.dev_idx]
        print(f"Connecting to {dev.product_id} ({dev.udid}) ...")

        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(dev)  # capture starts immediately

    # -------------------------------------------------------------------- #
    # Main loop
    # -------------------------------------------------------------------- #
    def run(self):
        self._open_device()

        # Spawn the Rerun viewer once
        rr.init("3d_vision_demo", spawn=True)

        recording_list = []

        counter = 0

        try:
            while True:
                self.event.wait()      # block until a frame arrives
                self.event.clear()

                # ----------- fetch the current RGB-D packet ---------------
                depth = self.session.get_depth_frame()
                rgb = self.session.get_rgb_frame()

                if depth is None or rgb is None:
                    continue  # corrupted packet - skip
                # rotate image counter-clockwise 90 degrees
                intr = self._intrinsic_matrix(self.session.get_intrinsic_mat())


                pose = self.session.get_camera_pose()                
                

                # ------------ log the current camera pose -----------------
                rr.log(
                    "camera",
                    rr.Transform3D(
                        translation=[pose.tx, pose.ty, pose.tz],
                        rotation=rr.Quaternion( xyzw=[pose.qx, pose.qy, pose.qz, pose.qw] ),
                    ),
                )

                print(f"Depth shape: {depth.shape}")
                print(f"RGB shape: {rgb.shape}")


                # up-scale depth to the RGB resolution if necessary
                if depth.shape[:2] != rgb.shape[:2]:
                    depth = cv2.resize(
                        depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST
                    )

                # --------------- build a point cloud ----------------------
                h, w = depth.shape
                # Cache grid indices to avoid recalculating every frame
                if not hasattr(self, "_cached_grid") or self._cached_grid.shape[1:] != (h, w):
                    self._cached_grid = np.indices((h, w), dtype=np.float32)
                ys, xs = self._cached_grid

                z = depth.astype(np.float32)
                valid = z > 0.0

                x_norm = (xs - intr[0, 2]) / intr[0, 0]
                y_norm = (ys - intr[1, 2]) / intr[1, 1]

                points = np.column_stack(
                    (x_norm[valid] * z[valid], y_norm[valid] * z[valid], z[valid])
                )
                colours = rgb[valid].astype(np.uint8).reshape(-1, 3)

                # downsample the point cloud to 1/4 of the original size
                points = points[::4]
                colours = colours[::4]

                rr.log("point_cloud", rr.Points3D(points, colors=colours))


                rgb = np.rot90(rgb, k=1)

                rr.log("rgb", rr.Image(rgb))

                # create a dictionary to hold the data

                pose_np = np.array([pose.tx, pose.ty, pose.tz, pose.qx, pose.qy, pose.qz, pose.qw])
                # get intrinsic matrix

                if counter % 6 == 0:

                    data = {
                        "depth": depth,
                        "rgb": rgb,
                        "pose": pose_np,
                        "intrinsic_matrix": intr,
                    }
                    # add the data to the recording list
                    recording_list.append(data)
                    counter = 0

                counter += 1

                
 

        finally:
            if self.session is not None:
                self.session.disconnect()
            cv2.destroyAllWindows()

            # record the data to a pkl file
            with open("recording.pkl", "wb") as f:
                import pickle
                pickle.dump(recording_list, f)
                print("Recording saved to recording.pkl")


if __name__ == "__main__":
    # Graceful ^C handling so Rerun viewer terminates correctly
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    DemoApp(dev_idx=0).run()
