# %%
import mujoco
import mujoco_viewer
import matplotlib.pyplot as plt
import numpy as np
import time
import mujoco.viewer
# Define the MuJoCo model using XML string
xml = """
<mujoco>
	<!-- <option gravity="0 0 0" /> -->
    <option gravity="0 0 -9.8" integrator="RK4" timestep="0.0001"/>
	<worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 10" dir="0 0 -1"/>
		<geom type="plane" size="5 0.2 0.1" rgba=".9 .9 .9 1" pos="0 0 0"/>
        <camera name="main" pos="0 -.1 .07" xyaxes="1 0 0 0 1 2"/>

		<body pos="0 0 1.5">
			<joint name="slide_joint" type="slide" axis="1 0 0"/>
			<geom type="box" size=".3 .2 .1" rgba="0.9 0 0.0 1" mass="0.5"/>

            <body pos="0 0 0">
                <joint name="hinge_joint" type="hinge" axis="0 -1 0" />

                <geom type="cylinder" size=".01 .5" rgba="0 0.9 0.9 1" pos="0 0 .5" mass="0"/>

                <body pos="0 0 1">
                    <geom type="sphere" size=".1" rgba="0 0.9 0 1" mass="2"/>
                </body>
            </body>
		</body>

	</worldbody>

    <keyframe>
        <key name="up" qpos="0 0.5" qvel="0 0.5" />
        <key name="down" qpos="0 3.14" qvel="0 0" />
    </keyframe>

    <actuator>
        <!-- Direct force on slide_joint: force = ctrl * gear -->
        <general
            name="slide_force"
            joint="slide_joint"
            ctrllimited="true"
            ctrlrange="-100 100"
            gear="1"
            gainprm="1 0 0"
            biasprm="0 0 0"/>
    </actuator>

</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

STEP_NUM = 50000
timevals = np.zeros(STEP_NUM)
slidevals = np.zeros(STEP_NUM)
hingevals = np.zeros(STEP_NUM)
controlvals = np.zeros(STEP_NUM)
# Params for controller
x = np.zeros(4) # state vector: [cart position, cart velocity, pole angle, pole angular velocity]
x_hist = np.zeros((STEP_NUM, 4))
# K = np.array([1.530612244827555,	2.346938775402242,	-42.530612242941501,	-7.346938775172230]) # Modal regulator
# K = np.array([6.717200085794616,	12.014432304358504,	-92.836436902761562,	-19.724418610963063]) # LMI regulator
K = np.array([0.999999999999973,	2.790405295241197,	-57.753172387381255,	-10.455467094560309])  # LQR regulator
# K = np.array([3.162277660168446,	5.231922504546993,	-65.365407167213959,	-13.244608041060054]) # LQR regulator

INTERACTIVE = True

# %%
if INTERACTIVE:
    with mujoco.viewer.launch_passive(model, data) as viewer:

        with viewer.lock():
            # Reset the simulation.
            mujoco.mj_resetDataKeyframe(model, data, 0)
            # Reset the free camera.
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)
            # Enable site frame visualization.
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
            viewer.opt.sitegroup[4] = 1
            viewer.opt.geomgroup[4] = 1

        while viewer.is_running():

            step_start = time.time()

            # Apply control
            x = np.array([data.joint("slide_joint").qpos[0],
                        data.joint("slide_joint").qvel[0],
                        data.joint("hinge_joint").qpos[0], 
                        data.joint("hinge_joint").qvel[0]])
            data.ctrl[0] = K @ x  # state feedback control law

            with viewer.lock():
                mujoco.mj_step(model, data)
                viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

else:

# %%

    try:
        mujoco.mj_deleteViewersForModel(model)
        del viewer
    except:
        pass

    viewer = mujoco_viewer.MujocoViewer(model, 
                                        data, 
                                        title="cartpole", 
                                        width=1920, 
                                        height=1080)

    # Change camera parameters
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.distance = 5.0       # zoom
    viewer.cam.azimuth = 90.0       # rotate around z
    viewer.cam.elevation = 0.0    # tilt up/down
    viewer.cam.lookat[:] = [0.0, 0.0, 2.0]  # point to look at

    # Reset the simulation to the first keyframe
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # Set speed of simulation
    RENDER_EVERY = 100

    for i in range(STEP_NUM):  
        if viewer.is_alive:     
            

            # Apply control
            x = np.array([data.joint("slide_joint").qpos[0],
                        data.joint("slide_joint").qvel[0],
                        data.joint("hinge_joint").qpos[0], # % (np.pi * 2) - np.pi,
                        data.joint("hinge_joint").qvel[0]])
            data.ctrl[0] = K @ x  # state feedback control law

            # Record data
            timevals[i] = data.time
            controlvals[i] = data.ctrl[0]
            x_hist[i, :] = x
            
            mujoco.mj_step(model, data)
            if i % RENDER_EVERY == 0:
                viewer.render()
                pass

        else:
            break
    viewer.close()
# %%

    dpi = 120
    width = 600
    height = 1200
    figsize = (width / dpi, height / dpi)
    _, ax = plt.subplots(5, 1, figsize=figsize, dpi=dpi, sharex=True)

    ax[0].plot(timevals, x_hist[:, 0])
    ax[0].set_ylabel('cart position (m)')
    _ = ax[0].set_title('cart position')

    ax[1].plot(timevals, x_hist[:, 1])
    ax[1].set_ylabel('cart velocity (m/s)')
    _ = ax[1].set_title('cart velocity')

    ax[2].plot(timevals, x_hist[:, 2])
    ax[2].set_ylabel('pole angle (rad)')
    _ = ax[2].set_title('pole angle')

    ax[3].plot(timevals, x_hist[:, 3])
    ax[3].set_ylabel('pole angular velocity (rad/s)')
    _ = ax[3].set_title('pole angular velocity')

    ax[4].plot(timevals, controlvals)
    ax[4].set_ylabel('control input')
    _ = ax[4].set_title('control input')

    ax[4].set_xlabel('time (seconds)')

    plt.tight_layout()
    plt.show()