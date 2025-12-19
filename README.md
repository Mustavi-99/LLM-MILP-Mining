# Open-Pit Mine Production Scheduler Simulation

`Simulator.py` is a Python-based simulation tool for Open-Pit Mine Production Scheduling. It models the extraction and processing decisions of mining blocks over multiple time periods to maximize Net Present Value (NPV).

The system facilitates interactive decision-making, allowing users to manually select blocks for extraction or processing while automatically validating compliance with physical slope constraints and processing capacities. The interface displays the real-time mine state and lists all currently valid actions in the format: (`action`, `block_id`, `remaining_amount`, `unit_npv`).

## Usage
Run the simulation from the terminal. You can customize the mine size and simulation duration using arguments.

### Basic Run (Default Settings)
Runs with 20 periods and a 3x3x3 block grid.

```bash
python Simulator.py
```
### Custom Configuration
Specify the number of periods and the dimensions of the mine.

```bash
python Simulator.py --num_periods 50 --x 10 --y 10 --z 5
```

### Interactive Controls
During the simulation, the system will prompt you for actions:

1. **Choose Action**:

    - `1` or `extract`: Mine a block from the pit.

    - `2` or `process`: Send a mined block to the processing plant.

2. **Select Block**: Enter the unique Integer ID of the block.

3. **Amount**: Enter the tonnage to process/extract.

The simulation ends when the final time period is reached or no profitable actions remain.

## Example Output

```
Initializing Simulation: Periods=20, Grid=(3, 3, 3)
Time Period 1
Current NPV: 0.00
Mine State:
[...Matrix Data...]
Possible Actions:
[(1, 1, 50, ), (1, 2, 50, )...]
Choose action ('extract/1' or 'process/2'): 1
Enter block number: 1
Enter amount: 50
Executing: (1, 1, 50)
```