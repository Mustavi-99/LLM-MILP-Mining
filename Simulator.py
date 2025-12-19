import argparse
from input_properties import Mine_properties, concentration_func
from mining_data import mining_data
from Mine import Mine
from Time import Time
from Scheduler import Scheduler


parser = argparse.ArgumentParser(description="Run the Mine Simulation Scheduler.")
parser.add_argument('--num_periods', type=int, default=20, help='Total simulation periods (default: 20)')
parser.add_argument('--x', type=int, default=3, help='Grid X dimension (default: 3)')
parser.add_argument('--y', type=int, default=3, help='Grid Y dimension (default: 3)')
parser.add_argument('--z', type=int, default=3, help='Grid Z dimension (default: 3)')

args = parser.parse_args()

print(f"Initializing Simulation: Periods={args.num_periods}, Grid=({args.x}, {args.y}, {args.z})")

state = mining_data((args.x, args.y, args.z))
arr, block_properties, HnD = state.build(concentration_func=concentration_func)

mining_instance = Mine(
    arr,
    block_properties,
    HnD,
    args.num_periods,
    Mine_properties
)

time_func = Time()
scheduler = Scheduler(mining_instance, time_func)

action_list = scheduler.possible_actions()
scheduler.update_action(action_list, action='initialize')

period = scheduler.time_func.period()

while period <= args.num_periods:
    
    print(f"\nTime Period {period}")
    try:
        print(f"Current NPV: {scheduler.mine.npv:.2f}")
    except:
            print(f"Current NPV: {scheduler.mine.npv}")
            
    print(f"Mine State:\n{scheduler.mine.mine_state()}")
    print(f"Possible Actions:\n{scheduler.possible_actions()}")

    while True:
        mapping = {
            "extract": 1, "extraction": 1, "1": 1,
            "process": 2, "processing": 2, "2": 2
        }
        while True:
            raw_choice = input("\nChoose action ('extract/1' or 'process/2'): ").strip().lower()
            if raw_choice in mapping:
                choice = mapping[raw_choice]
                break
            print("Invalid choice. Please enter '1', '2', 'extract', or 'process'.")

        while True:
            try:
                block_number = int(input("Enter block number: "))
                break
            except ValueError:
                print("Invalid input. Please enter a numeric block number.")

        while True:
            try:
                amount = int(input("Enter amount: "))
                break
            except ValueError:
                print("Invalid input. Please enter a numeric amount.")

        user_action = (choice, block_number, amount)
        current_possible = scheduler.possible_actions()

        is_valid = any(
            c == user_action[0] and b == user_action[1] and user_action[2] <= cap
            for c, b, cap, _ in current_possible
        )

        if is_valid:
            break
        else:
            print(f"Action {user_action} is not valid or exceeds capacity. Try again.")

    print(f"Executing: {user_action}")
    action_result, pv = scheduler.choosen_action(user_action)
    scheduler.update_action(action_result, present_value=pv)
    scheduler.action(action_result)

    if len(scheduler.possible_actions()) == 0:
        print("No more actions in this period. Advancing time...")
        scheduler.time_tick()
        period = scheduler.time_func.period()
        
        if len(scheduler.possible_actions()) == 0:
            print("No more possible actions available in future periods. Simulation Complete.")
            break
        else:
            scheduler.update_action(scheduler.possible_actions(), action='initialize')