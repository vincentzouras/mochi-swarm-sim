# type: ignore[reportAttributeAccessIssue]

import mujoco as mj
import os
import time
from src.simulation import Simulation
from src.controller import Controller

MODEL_XML_PATH = "models/mochi.xml"


def main(headless=False, inject_noise=False, noise_std = [0.0, 0.0, 0.0], duration=60.0):
    """
    Run the simulation in headless mode for fast data collection.
    
    Args:
        headless: If True, run without GUI (faster). If False, run with GUI.
        duration: Duration in seconds to run the simulation (only used in headless mode)
    """
    # 1. Load model and data
    model = mj.MjModel.from_xml_path(MODEL_XML_PATH)
    data = mj.MjData(model)

    # 2. Create the controller
    controller = Controller(model, data)

    # 3. Set up exploration state and arm the system (for headless mode)
    if headless:
        from src.state.exploration_state import ExplorationState
        from src.definitions import Action
        controller.state_machine.current_state = ExplorationState()
        controller.action_states[Action.ARMED] = True
        print("[HEADLESS MODE] ExplorationState enabled and ARMED")

    # 4. Create the simulation "engine" and pass it the controller
    sim = Simulation(model, data, controller, headless=headless)

    # 5. Enable noise injection for data collection (optional)
    # Adjust noise levels as needed: left_thrust_std, right_thrust_std, servo_angle_std
    if inject_noise:
        controller.enable_noise(
            left_thrust_std=noise_std[0],    # 10% noise on left motor
            right_thrust_std=noise_std[1],   # 10% noise on right motor
            servo_angle_std=noise_std[2],    # 0.1 rad (~5.7°) noise on servo
            seed=None               # Set to an integer for reproducible noise
        )
    
    # 6. Start data collection
    controller.start_data_collection()
    print(f"[DATA COLLECTION] Started. Running for {duration} seconds...")

    try:
        # 7. Run the simulation
        if headless:
            sim.run(duration=duration)
        else:
            sim.run()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Simulation stopped by user (Ctrl+C)")
    finally:
        # 8. Save the collected data (runs even if interrupted)
        if controller.collect_data:
            num_tuples = len(controller.data_buffer)
            # Create directory if it doesn't exist
            os.makedirs("noisy_training_data_60s", exist_ok=True)
            # Include noise info in filename if noise is enabled
            noise_suffix = "_noisy" if controller.noise_enabled else ""
            # filename = f"noisy_training_data_60s/noisy_spiral_{duration}s.pkl"

            filename = f"noisy_training_data/noisy_spiral_{noise_std[0]}_{noise_std[1]}_{noise_std[2]}_{duration}s.pkl"
            controller.save_collected_data(filename)
            print(f"[DATA COLLECTION] Collected {num_tuples} (state, action, next_state) tuples")
            if controller.noise_enabled:
                print(f"[NOISE] Data collected with noise injection enabled")
        else:
            print("No data was collected.")
        
        # Clean up: Clear control callback
        try:
            mj.set_mjcb_control(None)
        except:
            pass


if __name__ == "__main__":
    # Run in headless mode for fast data collection
    # Set headless=False to use GUI, or change duration to run for different time
    
    # Run with different noise levels from [0,0,0] to [0.5,0.5,1.5]
    # You can loop through different noise levels or run individually:
    
    # Example: Run with no noise
    # main(headless=True, inject_noise=True, noise_std=[0.0, 0.0, 0.0], duration=120.0)
    
    # Example: Run with moderate noise
    # main(headless=True, inject_noise=True, noise_std=[0.1, 0.1, 0.2], duration=120.0)
    
    # Example: Run with high noise (current setting)
    # main(headless=True, inject_noise=True, noise_std=[0.5, 0.5, 1.5], duration=120.0)
    
    # To run multiple noise levels in a loop:
    # Create directory before loop
    os.makedirs("noisy_training_data_60s", exist_ok=True)
    
    total_runs = 6 * 6 * 6  # 216 total runs
    run_count = 0
    
    for left_std in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        for right_std in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            for servo_std in [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]:
                run_count += 1
                print(f"\n{'='*60}")
                print(f"Run {run_count}/{total_runs}: noise_std=[{left_std}, {right_std}, {servo_std}]")
                print(f"{'='*60}")
                
                try:
                    # Clear any previous control callback before creating new model
                    try:
                        mj.set_mjcb_control(None)
                    except:
                        pass
                    
                    # Small delay to ensure cleanup
                    time.sleep(0.1)
                    
                    main(headless=True, inject_noise=True, 
                         noise_std=[left_std, right_std, servo_std], duration=60.0)
                    
                    # Clear control callback after run
                    try:
                        mj.set_mjcb_control(None)
                    except:
                        pass
                    
                except Exception as e:
                    import traceback
                    print(f"ERROR in run {run_count}: {e}")
                    print(traceback.format_exc())
                    # Clear control callback on error
                    try:
                        mj.set_mjcb_control(None)
                    except:
                        pass
                    continue
