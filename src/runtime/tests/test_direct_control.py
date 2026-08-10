import numpy as np
import pytest
from alpasim_grpc.v0.controller_pb2 import DirectControl
from alpasim_runtime.events.base import EventQueue
from alpasim_runtime.events.controller import ControllerEvent
from alpasim_runtime.events.policy import PolicyEvent
from alpasim_runtime.events.state import RolloutState, ServiceBundle
from alpasim_runtime.services.controller_service import ControllerService
from alpasim_utils.geometry import Pose


@pytest.mark.asyncio
async def test_policy_event_preserves_direct_control(
    rollout_state: RolloutState,
    service_bundle: ServiceBundle,
    mock_driver,
) -> None:
    """Store a physical driver command without trajectory transformation."""
    control = DirectControl(
        front_wheel_steering_angle_rad=0.2,
        longitudinal_acceleration_mps2=-1.5,
    )
    mock_driver.drive.return_value = (control, False)
    event = PolicyEvent(
        timestamp_us=200_000,
        policy_timestep_us=100_000,
        services=service_bundle,
        camera_ids=[],
        route_generator=None,
        send_recording_ground_truth=False,
    )

    await event.run(rollout_state, EventQueue())

    assert rollout_state.step_context is not None
    assert rollout_state.step_context.driver_trajectory is None
    assert rollout_state.step_context.direct_control == control


@pytest.mark.asyncio
async def test_controller_event_forwards_direct_control(
    rollout_state: RolloutState,
    service_bundle: ServiceBundle,
    mock_controller,
) -> None:
    """Forward direct control while deriving velocity from the current ego history."""
    control = DirectControl(
        front_wheel_steering_angle_rad=-0.1,
        longitudinal_acceleration_mps2=2.0,
    )
    assert rollout_state.step_context is not None
    rollout_state.step_context.step_start_us = 200_000
    rollout_state.step_context.target_time_us = 300_000
    rollout_state.step_context.force_gt = False
    rollout_state.step_context.driver_trajectory = None
    rollout_state.step_context.direct_control = control
    event = ControllerEvent(
        timestamp_us=200_000,
        control_timestep_us=100_000,
        services=service_bundle,
    )

    await event.run(rollout_state, EventQueue())

    kwargs = mock_controller.run_controller_and_vehicle.call_args.kwargs
    assert kwargs["direct_control"] == control
    assert kwargs["rig_reference_trajectory_in_rig"] is None
    assert kwargs["fallback_trajectory_local_to_rig"] is None
    assert np.isfinite(kwargs["rig_linear_velocity_in_rig"]).all()


def test_controller_request_selects_direct_control() -> None:
    """Serialize a direct command instead of a planned trajectory."""
    control = DirectControl(
        front_wheel_steering_angle_rad=0.3,
        longitudinal_acceleration_mps2=-2.0,
    )
    request = ControllerService.create_run_controller_and_vehicle_request(
        session_uuid="session",
        now_us=0,
        pose_local_to_rig=Pose.identity(),
        rig_linear_velocity_in_rig=np.zeros(3),
        rig_angular_velocity_in_rig=np.zeros(3),
        rig_reference_trajectory_in_rig=None,
        direct_control=control,
        future_us=100_000,
        force_gt=False,
    )

    assert request.WhichOneof("command") == "direct_control"
    assert request.direct_control == control


@pytest.mark.asyncio
async def test_force_gt_overrides_direct_control(
    rollout_state: RolloutState,
    service_bundle: ServiceBundle,
    mock_controller,
) -> None:
    """Keep recorded-GT warm-up behavior even if a driver returns direct control."""
    assert rollout_state.step_context is not None
    rollout_state.step_context.step_start_us = 200_000
    rollout_state.step_context.target_time_us = 300_000
    rollout_state.step_context.force_gt = True
    rollout_state.step_context.direct_control = DirectControl(
        front_wheel_steering_angle_rad=0.5,
        longitudinal_acceleration_mps2=6.0,
    )
    rollout_state.unbound.force_gt_period = range(0, 300_001)
    event = ControllerEvent(
        timestamp_us=200_000,
        control_timestep_us=100_000,
        services=service_bundle,
    )

    await event.run(rollout_state, EventQueue())

    kwargs = mock_controller.run_controller_and_vehicle.call_args.kwargs
    assert kwargs["direct_control"] is None
    assert kwargs["rig_reference_trajectory_in_rig"] is not None
    assert kwargs["fallback_trajectory_local_to_rig"] is not None
