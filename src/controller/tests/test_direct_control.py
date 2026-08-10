import math

import pytest
from alpasim_controller.mpc_controller import ControllerConfig
from alpasim_controller.system_manager import SystemManager
from alpasim_grpc.v0 import controller_pb2


def _request(
    session_uuid: str,
    steering_rad: float,
    acceleration_mps2: float,
) -> controller_pb2.RunControllerAndVehicleModelRequest:
    """Build one 100 ms direct-control propagation request."""
    request = controller_pb2.RunControllerAndVehicleModelRequest(
        session_uuid=session_uuid,
        future_time_us=100_000,
        pose_reporting_interval_us=25_000,
    )
    request.state.timestamp_us = 0
    request.state.pose.quat.w = 1.0
    request.state.state.linear_velocity.x = 5.0
    request.direct_control.front_wheel_steering_angle_rad = steering_rad
    request.direct_control.longitudinal_acceleration_mps2 = acceleration_mps2
    return request


def _run(
    tmp_path,
    session_uuid: str,
    steering_rad: float = 0.0,
    acceleration_mps2: float = 0.0,
):
    """Run one direct-control request in a fresh controller session."""
    manager = SystemManager(str(tmp_path), controller_config=ControllerConfig())
    manager.start_session(session_uuid)
    return manager.run_controller_and_vehicle_model(
        _request(session_uuid, steering_rad, acceleration_mps2)
    )


def test_direct_control_propagates_without_mpc(tmp_path) -> None:
    """Apply zero, acceleration, braking, and steering through the vehicle model."""
    coast = _run(tmp_path, "coast")
    accelerate = _run(tmp_path, "accelerate", acceleration_mps2=6.0)
    brake = _run(tmp_path, "brake", acceleration_mps2=-8.0)
    steer = _run(tmp_path, "steer", steering_rad=math.pi / 4)

    assert [state.timestamp_us for state in coast.states] == [
        25_000,
        50_000,
        75_000,
        100_000,
    ]
    assert accelerate.states[-1].dynamic_state.linear_velocity.x > coast.states[
        -1
    ].dynamic_state.linear_velocity.x
    assert brake.states[-1].dynamic_state.linear_velocity.x < coast.states[
        -1
    ].dynamic_state.linear_velocity.x
    assert abs(steer.states[-1].pose_local_to_rig.quat.z) > 0.0


@pytest.mark.parametrize(
    "steering_rad,acceleration_mps2",
    [
        (math.pi / 4 + 0.01, 0.0),
        (0.0, 6.01),
        (0.0, -8.01),
        (math.nan, 0.0),
    ],
)
def test_direct_control_rejects_invalid_commands(
    tmp_path,
    steering_rad: float,
    acceleration_mps2: float,
) -> None:
    """Reject non-finite or out-of-range physical commands before propagation."""
    manager = SystemManager(str(tmp_path), controller_config=ControllerConfig())
    manager.start_session("invalid")

    with pytest.raises(ValueError):
        manager.run_controller_and_vehicle_model(
            _request("invalid", steering_rad, acceleration_mps2)
        )
