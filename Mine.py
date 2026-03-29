from xml.parsers.expat import model
import plotly.graph_objects as go
from gurobipy import Model, GRB
import gurobipy as gp
import math
from collections import defaultdict 

class HalfHourlyFileLogger:
    def __init__(self, filepath, interval=1800):
        self.interval = interval
        self.next_time = interval
        self.file = open(filepath, "a", buffering=1)

        self.file.write("time_sec,time_hr,obj,bound,gap\n")

    def __call__(self, model, where):
        if where == GRB.Callback.MIP:
            runtime = model.cbGet(GRB.Callback.RUNTIME)

            if runtime >= self.next_time:
                try:
                    obj = model.cbGet(GRB.Callback.MIP_OBJBST)
                    bound = model.cbGet(GRB.Callback.MIP_OBJBND)

                    # compute gap safely
                    if obj == 0 or obj == GRB.INFINITY or obj == -GRB.INFINITY:
                        gap = math.inf
                    else:
                        gap = abs(obj - bound) / abs(obj)

                    self.file.write(
                        f"{runtime:.1f},"
                        f"{runtime/3600:.3f},"
                        f"{obj},"
                        f"{bound},"
                        f"{gap}\n"
                    )
                    self.file.flush()

                except gp.GurobiError:
                    pass

                self.next_time += self.interval

    def close(self):
        self.file.close()

class Mine:
    def __init__(self, arr, block_properties, HnD, num_periods, mine_properties):
        self.arr = arr
        self.block_properties = block_properties
        self.HnD = HnD
        
        # Dimensions
        self.num_x = arr.shape[2]
        self.num_y = arr.shape[1]
        self.num_z = arr.shape[0]
        self.num_blocks = self.num_x * self.num_y * self.num_z  # Calculate once
        self.num_periods = num_periods
        
        # Financials
        self.discount_rate = mine_properties.get("discount_rate", 0.1)
        
        # --- 1. Block Tonnage Setup ---
        raw_block_tonnage = mine_properties.get("block_tonnage", None)
        
        if raw_block_tonnage is not None and isinstance(raw_block_tonnage, (int, float)):
            # Case A: Single global tonnage provided
            self.block_tonnage = [raw_block_tonnage] * self.num_blocks
        elif raw_block_tonnage is None:
            # Case B: Derive from block volume (default)
            self.block_tonnage = [v["volume"] * 100 for v in block_properties.values()]
        else:
            # Case C: List provided (assume valid length)
            self.block_tonnage = raw_block_tonnage

        # Add dummy index 0
        self.block_tonnage = [0] + self.block_tonnage

        # --- 2. Ore Tonnage Setup ---
        raw_ore_tonnage = mine_properties.get("ore_tonnage", None)

        if raw_ore_tonnage is not None:
            # If list is too short, repeat it; otherwise take exact length
            if len(raw_ore_tonnage) != self.num_blocks:
                expanded_ore = raw_ore_tonnage * self.num_blocks
                self.ore_tonnage = expanded_ore[:self.num_blocks]
            else:
                self.ore_tonnage = raw_ore_tonnage
        else:
            # Derive from Concentration * Block Tonnage
            # Note: block_tonnage has a 0 at index 0, so we slice it [1:] to match blocks
            self.ore_tonnage = [
                int(v["concentration"] * t) 
                for v, t in zip(block_properties.values(), self.block_tonnage[1:])
            ]

        # Add dummy index 0
        self.ore_tonnage = [0] + self.ore_tonnage

        # --- 3. Economics ---
        # Calculated per block based on tonnages
        self.revenue = [mine_properties.get("revenue", 40) * ton for ton in self.ore_tonnage]
        self.cost = [mine_properties.get("cost", 3) * ton for ton in self.block_tonnage]

        # --- 4. Capacity Constraints (Per Period) ---
        # Helper to create limit lists [0, Limit, Limit, ...]
        def create_limit_list(key):
            val = mine_properties.get(key, None)
            
            # FIX: If the value comes wrapped in a list (e.g. [6.0]), extract the number
            if isinstance(val, list) and len(val) > 0:
                val = val[0]
                
            return [0] + [val] * self.num_periods

        self.Mining_capacity_lower = create_limit_list("Mining_capacity_lower")
        self.Mining_capacity_upper = create_limit_list("Mining_capacity_upper")
        self.Mining_capacity_used = [0] * len(self.Mining_capacity_lower)

        self.Processing_capacity_lower = create_limit_list("Processing_capacity_lower")
        self.Processing_capacity_upper = create_limit_list("Processing_capacity_upper")
        self.Processing_capacity_used = [0] * len(self.Processing_capacity_lower)

        # --- 5. Grade Constraints (Per Period) ---
        # Note: Head_grade_lower/upper in properties is likely a scalar limit (e.g., 1.5)
        # We multiply a list containing that scalar: [val] * periods

        self.Head_grade_lower = [0] + mine_properties.get("Head_grade_lower", None) * (num_periods)
        self.Head_grade_upper = [0] + mine_properties.get("Head_grade_upper", None) * (num_periods)

        self.Head_grade = [v["concentration"] for v in block_properties.values()]
            
        self.Head_grade = [0] + self.Head_grade
        
        # --- Simulation State ---
        self.constraint_control = mine_properties.get("constraint_control", [0])
        self.mine_progress = mine_properties.get("progress")
        
        # State tracking arrays (Floats)
        self.mine_extraction = [0.0] * len(self.block_tonnage)
        self.mine_process = [0.0] * len(self.block_tonnage)
        
        # Copy initial tonnages to tracking lists
        self.To_be_extracted = self.block_tonnage.copy()
        self.To_be_processes = self.ore_tonnage.copy()
        
        # Blend tracking: Index 0 is dummy, Indices 1..T are empty lists
        self.grade_blend = [0] + [[] for _ in range(num_periods)]
        
        self.npv = 0
        
    def action_list(self, time_period=1):
        actions = []
        
        # Check if simulation time is over
        if self.num_periods < time_period:
            return actions

        # Optimization: Use a set for fast O(1) lookups
        done_blocks = {block for period in self.mine_progress for block in period}

        # --- BLOCK PRECEDENCE Constraints ---
        available_blocks = [
            k for k, deps in self.HnD.items()
            if not deps or all(d in done_blocks for d in deps)
        ]

        # --- EXTRACTION Constraints and variable  ---
        available_blocks_for_extract = [
            b for b in available_blocks if self.mine_extraction[b] < 1.0
        ]

        remaining_mining_cap = self.Mining_capacity_upper[time_period] - self.Mining_capacity_used[time_period]

        if remaining_mining_cap > 0:
            for block in available_blocks_for_extract:
                amount_to_extract = min(self.To_be_extracted[block], remaining_mining_cap)
                
                if amount_to_extract <= 0:
                    continue

                action = (1, block, amount_to_extract)
                # Use max possible extraction to determine unit value
                npv_total = self.calculate_NPV(action, time_period)
                value_per_unit = round(npv_total / amount_to_extract, 2) if amount_to_extract > 0 else 0
                
                # Extraction actions usually just take the amount
                actions.append((1, block, amount_to_extract, value_per_unit))

        # --- PROCESSING Constraints ---
        available_blocks_for_process = [
            b for b in available_blocks 
            if self.mine_extraction[b] == 1.0 and self.mine_process[b] < 1.0
        ]

        remaining_process_cap = self.Processing_capacity_upper[time_period] - self.Processing_capacity_used[time_period]

        if remaining_process_cap > 0:
            for block in available_blocks_for_process:
                
                # 1. Determine the physical maximum (Inventory vs Plant Capacity)
                physical_limit = min(self.To_be_processes[block], remaining_process_cap)
                
                # 2. Get Blending Limits
                # Returns: (min_required_amount, max_allowed_amount, feasible)
                min_blend, max_blend, feasible = self.grade_blending(block, time_period, physical_limit)
                
                # 3. Validation
                if not feasible:
                    continue
                
                if max_blend <= 0:
                    continue
                
                if min_blend > max_blend:
                    continue

                action_for_calc = (2, block, max_blend)
                npv_total = self.calculate_NPV(action_for_calc, time_period)
                value_per_unit = round(npv_total / max_blend, 2) if max_blend > 0 else 0

                actions.append((2, block, min_blend, max_blend, value_per_unit))

        return actions 
    # --- GRADE BLENDING Constraints ---
    def grade_blending(self, block_id, time_period, possible_processing_amount=None):
        
        if possible_processing_amount is None:
            possible_processing_amount = float('inf')
            
        g_block = self.Head_grade[block_id]
        target_min = self.Head_grade_lower[time_period]
        target_max = self.Head_grade_upper[time_period]
        
        
        if not self.grade_blend[time_period]:
            if target_min <= g_block <= target_max:
                return 0.0, possible_processing_amount, True
            else:
                return 0.0, 0.0, False

        current_mass = sum([amt for amt, conc in self.grade_blend[time_period]])
        current_metal = sum([conc * amt for amt, conc in self.grade_blend[time_period]])
        
        current_grade = current_metal / current_mass if current_mass > 0 else 0

        valid_min_tonnage = 0.0
        valid_max_tonnage = possible_processing_amount

        if g_block < target_min:
            if current_grade <= target_min:
                valid_max_tonnage = 0.0
            else:
                limit = (target_min * current_mass - current_metal) / (g_block - target_min)
                valid_max_tonnage = min(valid_max_tonnage, limit)
                
        elif g_block > target_min:
            if current_grade < target_min:
                required_amount = (target_min * current_mass - current_metal) / (g_block - target_min)
                valid_min_tonnage = max(valid_min_tonnage, required_amount)

        
        if g_block > target_max:
            if current_grade >= target_max:
                valid_max_tonnage = 0.0
            else:
                limit = (target_max * current_mass - current_metal) / (g_block - target_max)
                valid_max_tonnage = min(valid_max_tonnage, limit)
                
        elif g_block < target_max:
            if current_grade > target_max:
                required_amount = (target_max * current_mass - current_metal) / (g_block - target_max)
                valid_min_tonnage = max(valid_min_tonnage, required_amount)

        if valid_min_tonnage > valid_max_tonnage:
            return 0.0, 0.0, False
        else:
            return valid_min_tonnage, valid_max_tonnage, True
        
    # --- NPV Calculation ---    
    def calculate_NPV(self,action,t,c = None):
        choice, block_id, amount = action
        vt = self.revenue[block_id]
        q = self.cost[block_id]
        r = self.discount_rate
        if choice == 1:
            cost = q*amount/self.block_tonnage[block_id]
            return -(cost / ((1 + r) ** t))
        else:
            if self.ore_tonnage[block_id]==0:
                return 0
            revenue = vt * amount/self.ore_tonnage[block_id]
            return (revenue / ((1 + r) ** t))
        
    def update_npv(self,pv):
        # print(f"Updating NPV: {self.npv} = {pv}")
        self.npv += round(pv,2)
        return self.npv    
    
    def update(self, action,time_func):
        time_period = time_func.period()
        choice, block_id, amount = action
        
        if choice == 'volume':
            ton = self.block_tonnage[block_id]
            if ton <= 0:
                # print(f"Block {block_id} has zero tonnage.")
                return
            self.mine_extraction[block_id] += amount / ton
            self.To_be_extracted[block_id] = max(0.0, self.To_be_extracted[block_id] - amount)
            # protect index
            if 0 <= time_period < len(self.Mining_capacity_used):
                self.Mining_capacity_used[time_period] += amount

            if self.mine_extraction[block_id] >= 1.0 - 1e-4:
                self.mine_extraction[block_id] = 1.0
                if all(block_id not in row for row in self.mine_progress):
                    while len(self.mine_progress)<=time_period:
                        self.mine_progress.append([])
                    self.mine_progress[time_period].append(block_id)

        elif choice == 'concentration':
            ton = self.ore_tonnage[block_id]
            if ton <= 0:
                print(f"Block {block_id} has zero ore tonnage.")
                return
            self.mine_process[block_id] += round(amount / ton, 4)
            self.To_be_processes[block_id] = max(0.0, self.To_be_processes[block_id] - amount)
            if 0 <= time_period < len(self.Processing_capacity_used):
                self.Processing_capacity_used[time_period] += amount

            if self.mine_process[block_id] >= 1.0 - 1e-4:
                self.mine_process[block_id] = 1.0
            # print(f"Processing Successful. Processed: {round(self.mine_process[block_id]*100,2)}%. Capacity Used: {self.Processing_capacity_used[time_period]}")
            
        else:
            print("Unknown update action choice. Use 'volume' or 'concentration'.")

    
    def display_npv(self,time_period):
        print(f"Net Present value of time period {time_period} is: {self.npv}")

    
    
    
        
       
    def run_deterministic_milp2(self, number_of_periods: int | None = None):
        print("Running deterministic MILP...")
        """Build and solve a deterministic MILP for block mining."""
        # ---------- Basic data ----------
        num_blocks = self.num_x * self.num_y * self.num_z
        num_periods = number_of_periods or self.num_periods

        constraint_control = self.constraint_control
        r = self.discount_rate

        tt = self.block_tonnage or [0, 100, 100, 100]          # total tonnes per block
        o  = self.ore_tonnage   or [0, 60, 50, 40]            # ore tonnage per block
        w  = [tt[i] - o[i] for i in range(min(len(tt), len(o)))]  # waste tonnage per block

        vt = self.revenue or 10                                 # revenue per tonne
        q  = self.cost   or 5                                   # cost per tonne

        # Capacity and grade bounds (1‑based indexing)
        Cl = self.Mining_capacity_lower or [10] * (num_periods + 1)
        Cu = self.Mining_capacity_upper or [50] * (num_periods + 1)
        Ql = self.Processing_capacity_lower or [20] * (num_periods + 1)
        Qu = self.Processing_capacity_upper or [100] * (num_periods + 1)
        G_lb = self.Head_grade_lower or [0.4] * (num_periods + 1)
        G_ub = self.Head_grade_upper or [6.0] * (num_periods + 1)
        G = self.Head_grade or [4, 0.6, 2] * (num_blocks + 1)

        H_nD = self.HnD or defaultdict(list)

        # ---------- Model ----------
        model = gp.Model("deterministic_milp")
        # model.Params.OutputFlag = 0

        # ---------- Variables ----------
        x, y, b, u, l = (dict() for _ in range(5))

        for n in range(1, num_blocks + 1):
            for t in range(1, num_periods + 1):
                x[(n, t)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1,
                                         name=f"x_{n}_{t}")
                y[(n, t)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1,
                                         name=f"y_{n}_{t}")
                b[(n, t)] = model.addVar(vtype=GRB.BINARY,
                                         name=f"b_{n}_{t}")
                u[(n, t)] = model.addVar(vtype=GRB.BINARY,
                                         name=f"u_{n}_{t}")

        for t in range(1, num_periods + 1):
            l[t] = model.addVar(vtype=GRB.BINARY, name=f"l_{t}")

        model.update()

        # ---------- Objective ----------
        model.setObjective(
            gp.quicksum((vt[n] * x[(n, t)] - q[n] * y[(n, t)]) / (1 + r) ** t
                         for n in range(1, num_blocks + 1)
                         for t in range(1, num_periods + 1)),
            GRB.MAXIMIZE
        )

        # ---------- Constraints ----------
        # 1 – Mining capacity
        if 1 in constraint_control:
            block_remaining = {(n, t): (o[n] + w[n]) * (1 - gp.quicksum(y[(n, j)] for j in range(1, t)))
                               for n in range(1, num_blocks + 1)
                               for t in range(1, num_periods + 1)}

            block_total_remaining = {t: gp.quicksum(block_remaining[(n, t)] for n in range(1, num_blocks + 1))
                                     for t in range(1, num_periods + 1)}

            M = 1e6
            for t in range(1, num_periods + 1):
                model.addConstr(gp.quicksum((o[n] + w[n]) * y[(n, t)] for n in range(1, num_blocks + 1)) <= Cu[t])

                model.addConstr(l[t] <= 1 + (block_total_remaining[t] - Cl[t]) / M)
                model.addConstr(l[t] >= (block_total_remaining[t] - Cl[t]) / M)

                model.addConstr(gp.quicksum((o[n] + w[n]) * y[(n, t)] for n in range(1, num_blocks + 1)) >= Cl[t] * l[t])

        # 2 – Processing capacity
        if 2 in constraint_control:
            for t in range(1, num_periods + 1):
                model.addConstr(gp.quicksum(o[n] * x[(n, t)] for n in range(1, num_blocks + 1)) <= Qu[t])
                model.addConstr(gp.quicksum(o[n] * x[(n, t)] for n in range(1, num_blocks + 1)) >= Ql[t])

        # 3 – Grade constraints
        if 3 in constraint_control:
            for t in range(1, num_periods + 1):
                model.addConstr(gp.quicksum((G[n] - G_ub[t]) * o[n] * x[(n, t)]
                                            for n in range(1, num_blocks + 1)) <= 0)
                model.addConstr(gp.quicksum((G[n] - G_lb[t]) * o[n] * x[(n, t)]
                                            for n in range(1, num_blocks + 1)) >= 0)

        # 4 – Block precedence
        if 4 in constraint_control:
            for n in range(1, num_blocks + 1):
                for t in range(1, num_periods + 1):
                    for d in H_nD.get(n, []):
                        model.addConstr(b[(n, t)] - gp.quicksum(y[(d, j)] for j in range(1, t + 1)) <= 0)

            for n in range(1, num_blocks + 1):
                for t in range(1, num_periods + 1):
                    model.addConstr(gp.quicksum(y[(n, i)] for i in range(1, t + 1)) - b[(n, t)] <= 0)

            for n in range(1, num_blocks + 1):
                for t in range(1, num_periods):
                    model.addConstr(b[(n, t)] - b[(n, t + 1)] <= 0)

        # 5 – Variable control
        if 5 in constraint_control:
            for t in range(1, num_periods + 1):
                for n in range(1, num_blocks + 1):
                    model.addConstr(x[(n, t)] <= gp.quicksum(y[(n, j)] for j in range(1, t + 1)))

        # 6 – Mining‑time variable limits
        if 6 in constraint_control:
            for n in range(1, num_blocks + 1):
                # Sum of all ore‑time variables for the block
                sum_y_block = gp.quicksum(y[(n, j)] for j in range(1, num_blocks + 1))
                for t in range(1, num_periods + 1):
                    model.addConstr(u[(n, t)] <= sum_y_block)
                    model.addConstr(u[(n, t)] <= gp.quicksum(y[(n, j)] for j in range(1, t + 1)))
                    model.addConstr(x[(n, t)] <= u[(n, t)])

            for n in range(1, num_blocks + 1):
                model.addConstr(gp.quicksum(y[(n, j)] for j in range(1, num_blocks + 1)) <= 1)
                model.addConstr(gp.quicksum(y[(n, j)] for j in range(1, num_blocks + 1)) <= 1)

        # 7 – MILP gap & time limits for large problems
        if num_blocks > 20:
            model.setParam(GRB.Param.MIPGap, 0.05)
            model.setParam(GRB.Param.TimeLimit, 18000)

        # ---------- Solve ----------
        logger = HalfHourlyFileLogger("gurobi_progress.txt", interval=1800)
        model.optimize(logger)
        logger.close()

        return model, y, x, b, o, tt, Cl, l, Cu, Ql, Qu

    def run_deterministic_milp(self,number_of_periods=None):
        num_blocks = self.num_x * self.num_y * self.num_z
        if number_of_periods is not None:
            num_periods = number_of_periods
        else:
            num_periods = self.num_periods

        constraint_control = self.constraint_control
        r = self.discount_rate
        tt = self.block_tonnage if self.block_tonnage is not None else [0, 100, 100, 100]   # total tonnes per block
        o = self.ore_tonnage if self.ore_tonnage is not None else [0, 60, 50, 40]    # ore tonnage in block
        # print(len(tt),tt, len(o),o)
        w = [tt[i] - o[i] for i in range(min(len(tt), len(o)))]  # waste tonnage in block
        # print("Ore:", o)
        # print("Waste:", w)

        vt = self.revenue if self.revenue is not None else 10  # revenue per tonne
        q = self.cost if self.cost is not None else 5  # cost per tonne

        Cl = self.Mining_capacity_lower if self.Mining_capacity_lower is not None else [10] * (num_periods + 1)  # lower bound of mining capacity
        Cu = self.Mining_capacity_upper if self.Mining_capacity_upper is not None else [50] * (num_periods + 1)  # upper bound of mining capacity
        Ql = self.Processing_capacity_lower if self.Processing_capacity_lower is not None else [20] * (num_periods + 1)   # lower bound of processing capacity
        Qu = self.Processing_capacity_upper if self.Processing_capacity_upper is not None else [100] * (num_periods + 1)  # upper bound of processing capacity
        G_lb = self.Head_grade_lower if self.Head_grade_lower is not None else [0.4] * (num_periods + 1)  # lower bound of required head grade
        G_ub = self.Head_grade_upper if self.Head_grade_upper is not None else [6.0] * (num_periods + 1)  # upper bound of required head grade
        G = self.Head_grade if self.Head_grade is not None else [4,0.6,2] * (num_blocks + 1)
        

        model = Model("deterministic_milp")
        # model.Params.OutputFlag = 0
        
        H_nD = self.HnD
        if H_nD is None:
            H_nD = [[0] * (num_blocks + 1) for _ in range(num_blocks + 1)]
        
        x, y, b, u, l =  {}, {}, {}, {}, {}
        k ={}
        for n in range(1,num_blocks+1):
            for t in range(1,num_periods+1):
                x[n, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"x_{n}_{t}")
                y[n, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"y_{n}_{t}")
                b[n, t] = model.addVar(vtype=GRB.BINARY, name=f"b_{n}_{t}")
                u[n,t] = model.addVar(vtype=GRB.BINARY, name=f"u_{n}_{t}")
                
        for t in range(1,num_periods+1):
            l[t] = model.addVar(vtype=GRB.BINARY, name=f"l_{t}")
             

        model.update()
        # Objective function
        model.setObjective(
            gp.quicksum((vt[n] * x[n, t] - q[n] * y[n, t])  / (1 + r)**t 
                for n in range(1,num_blocks+1)
                for t in range(1,num_periods+1)),
            GRB.MAXIMIZE
        )

        if 1 in constraint_control:
            block_remaining = {}
            block_total_remaining = {}

            # ore_remaining = {}
            # ore_total_remaining = {}
            # print(f"Number of Blocks: {num_blocks}, Number of Periods: {num_periods}, o: {len(o)}, w: {len(w)}")
            for n in range(1, num_blocks + 1):
                for t in range(1, num_periods+1):
                    expr_remain = (o[n] + w[n]) * (1 - gp.quicksum(y[n, j] for j in range(1, t)))
                    block_remaining[n, t] = expr_remain
                    
                    
            M = 1e6
            eps = 1e-6
            for t in range(1, num_periods+1):
                # Build total remaining expression at time t
                expr_total = gp.quicksum(block_remaining[n, t] for n in range(1, num_blocks + 1))
                block_total_remaining[t] = expr_total
                

            for t in range(1,num_periods+1):
                model.addConstr(gp.quicksum((o[n] + w[n]) * y[n, t] for n in range(1,num_blocks+1)) <= Cu[t])
                
                model.addConstr(l[t]<=(1+(block_total_remaining[t]-Cl[t])/M))
                model.addConstr(l[t]>=(block_total_remaining[t]-Cl[t])/M)

                model.addConstr(gp.quicksum((o[n] + w[n]) * y[n, t] for n in range(1, num_blocks + 1))>= Cl[t] * l[t])
               
        # Processing capacity
        if 2 in constraint_control:
            for t in range(1,num_periods+1):
                model.addConstr(gp.quicksum(o[n] * x[n, t] for n in range(1,num_blocks+1)) <= Qu[t])
                
                model.addConstr(gp.quicksum(o[n] * x[n, t] for n in range(1,num_blocks+1)) >= Ql[t])

        
        # Grade constraints ###################
        if 3 in constraint_control:
            # print(f"{len(G)},{len(o)}, {len(G_ub)}, {len(G_lb)}")
            for t in range(1,num_periods+1):
                # print(f"Adding grade constraints for period {t}")
                # print(f"G_ub[{t}]: {G_ub[t]}, G_lb[{t}]: {G_lb[t]}")
                model.addConstr(gp.quicksum((G[n]-G_ub[t])*o[n]*x[n,t] for n in range(1,num_blocks+1)) <=0)
                model.addConstr(gp.quicksum((G[n]-G_lb[t])*o[n]*x[n,t] for n in range(1,num_blocks+1)) >=0)
        
        # Block precedence constraints #########################
        if 4 in constraint_control:
            for n in range(1, num_blocks + 1):
                for t in range(1, num_periods + 1):
                    for d in H_nD.get(n, []):
                        model.addConstr(b[n, t] - gp.quicksum(y[d, j] for j in range(1, t + 1)) <= 0)
                    
            for n in range(1,num_blocks+1):
                for t in range(1,num_periods+1):
                    model.addConstr(gp.quicksum(y[n, i] for i in range(1, t+1)) - b[n, t] <= 0)
                    
            for n in range(1,num_blocks+1):
                for t in range(1,num_periods): 
                    model.addConstr(b[n, t] - b[n, t + 1] <= 0) 
                    
        # Variable control constraints
        if 5 in constraint_control:
            for t in range(1,num_periods+1):
                for n in range(1,num_blocks+1):
                    model.addConstr(x[n, t] <= gp.quicksum(y[n, j] for j in range(1,t+1)))
                
        #######################################
        if 6 in constraint_control:
            for n in range(1,num_blocks+1):
                model.addConstr(gp.quicksum(x[n, j] for j in range(1, num_periods + 1)) <= 1)
                model.addConstr(gp.quicksum(y[n, j] for j in range(1, num_periods + 1)) <= 1)
                for t in range(1,num_periods+1):
                    # model.addConstr(u[n,t] <= gp.quicksum(y[n, j] for j in range(1,num_periods+1)))
                    model.addConstr(u[n,t] <= gp.quicksum(y[n, j] for j in range(1,t+1)))
                    model.addConstr(x[n,t] <= u[n,t])

        if num_blocks > 20:
            model.setParam("MIPGap", 0.05)
            # model.setParam("TimeLimit", 18000)
                    
        # model.write("deterministic_model.lp") 
        # model.optimize()
        logger = HalfHourlyFileLogger(f"{num_blocks}_p_{num_periods}_gurobi_progress.txt", interval=1800)

        model.optimize(logger)

        logger.close()
        # print("------------------------------------------------------\n")
            
        return model,y,x,b,o,tt,Cl,l,Cu,Ql,Qu    
    
    def mine_state(self):
        state = [
                    [
                        [
                            (b.item(),
                            self.block_properties.get(b)['concentration'],
                            self.To_be_extracted[b],
                            self.To_be_processes[b])
                            for b in row
                        ]
                        for row in block
                    ]
                    for block in self.arr
                ]
        return state
    
    def print_info(self):
        print("Mine Information:")
        print(f"Dimensions: {self.num_x} x {self.num_y} x {self.num_z}")
        print(f"Number of Periods: {self.num_periods}")
        print(f"Discount Rate: {self.discount_rate}")
        print(f"Block Tonnage: {self.block_tonnage[1:]}")
        print(f"Ore Tonnage: {self.ore_tonnage[1:]}")
        print(f"Revenue per unit: {self.revenue}")
        print(f"Cost per unit: {self.cost}")
        print(f"Mining Capacity Lower Bounds: {self.Mining_capacity_lower[1:]}")
        print(f"Mining Capacity Upper Bounds: {self.Mining_capacity_upper[1:]}")
        print(f"Processing Capacity Lower Bounds: {self.Processing_capacity_lower[1:]}")
        print(f"Processing Capacity Upper Bounds: {self.Processing_capacity_upper[1:]}")
        print(f"Head Grade Lower Bounds: {self.Head_grade_lower[1:]}")
        print(f"Head Grade Upper Bounds: {self.Head_grade_upper[1:]}")
        print(f"Head Grade: {self.Head_grade[1:]}")
        print(f"Constraint Control: {self.constraint_control[1:]}")