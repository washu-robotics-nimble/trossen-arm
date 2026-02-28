# https://docs.trossenrobotics.com/trossen_arm/main/getting_started/demo_scripts.html 
# Import the driver
import trossen_arm

if __name__ == "__main__":
    # Create a driver object
    driver = trossen_arm.TrossenArmDriver()

    # Configure the driver
    driver.configure(...)

    # Beginning of an action

    #     Get the modes of all joints if needed
    #     Here xxxs are the modes of all the joints where xxx can be
    #     - trossen_arm.Mode.position
    #     - trossen_arm.Mode.velocity
    #     - trossen_arm.Mode.external_effort
    #     - trossen_arm.Mode.effort
    xxxs = driver.get_modes()

    #     Set the mode[s] of the joint[s]
    #     Here yyy can be arm, gripper, all, or joint where
    #     - all includes all the joints
    #     - arm includes all joints but the gripper joint
    #     - gripper includes just the gripper joint
    #     - joint includes a specific zero-indexed joint
    driver.set_yyy_mode[s](xxx)

    #     Start moving the joint[s]

    #         Some logic

    #         Command the joint[s]
    #
    #         A joint command includes
    #         - goal[s]
    #         - time to reach the goal[s]
    #         - whether to block until reaching goal[s]
    #         - optionally the goal derivative[s]
    #         where yyy and zzz must be compatible with the mode set above
    #
    #         Alternatively, if the arm joints all have one of the following modes
    #         - trossen_arm.Mode.position
    #           pose of the tool frame measured in the base frame
    #         - trossen_arm.Mode.velocity
    #           linear and angular velocities of the tool frame measured in the base frame
    #         - trossen_arm.Mode.external_effort
    #           linear and angular efforts to be applied at the tool frame
    #           measured in the base frame while compensating for gravity and friction
    #         We can also command the arm joints to move in Cartesian space
    #         The Cartesian command includes an additional argument: interpolation space
    #         - trossen_arm.InterpolationSpace.joint
    #           Interpolate from start to goal state in joint space
    #         - trossen_arm.InterpolationSpace.cartesian
    #           Interpolate from start to goal state in Cartesian space
    driver.set_yyy_zzz[s](...) | driver.set_cartesian_zzzs(...)

    #         Get the robot outputs if needed
    #         The robot output includes
    #         - header
    #           - id
    #           - timestamp
    #         - joint space states
    #           - positions
    #           - velocities
    #           - external_efforts
    #           - efforts
    #           - compensation_efforts
    #           - rotor_temperatures
    #           - driver_temperatures
    #         - Cartesian space states
    #           - positions
    #           - velocities
    #           - external_efforts
    robot_output: trossen_arm.RobotOutput = driver.get_robot_output()

    #         Some more logic

    #     Stop moving the joint[s]

    # End of an action

    # More actions if needed