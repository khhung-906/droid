import argparse

import zerorpc

from droid.franka.robot import FrankaRobot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=4242, help="zerorpc bind port for this arm")
    parser.add_argument("--robot-port", type=int, default=None,
                        help="polymetis RobotInterface port (per-arm when 2 controllers share a NUC)")
    parser.add_argument("--gripper-port", type=int, default=None,
                        help="polymetis GripperInterface port (per-arm when 2 grippers share a NUC)")
    args = parser.parse_args()

    robot_client = FrankaRobot(robot_port=args.robot_port, gripper_port=args.gripper_port)
    s = zerorpc.Server(robot_client)
    s.bind(f"tcp://0.0.0.0:{args.port}")
    s.run()
