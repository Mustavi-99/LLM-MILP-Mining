import pandas as pd
import random
from gurobipy import Model, GRB
class Scheduler:
    def __init__(self, mine,time_func):
        self.mine = mine
        self.time_func = time_func
        self.actions_taken = {}
        self.extraction_portions = {}
        self.processing_portions = {}
        self.npv = 0
        self.check_process=0
        self.choice = 'random'

    def schedule(self,choice='random'):
        choice = choice
        self.choice = choice
        action_list = self.possible_actions()
        work_number = 0
        # for actions in action_list:
        #     print(f"{actions}")
        # print(f"\n{choice} scheduling starts.")
        self.update_action(action_list,action='initialize')
        # print(f"Scheduling startsPOIJ.")
        while True:
            if len(action_list):
                selected_action,pv = self.action_choice(action_list=action_list,choice = choice)
                self.update_action(selected_action,present_value=pv)
                if(self.action(selected_action)):
                    work_number += 1
                action_list = self.possible_actions()    
            else:
                self.time_tick()
                # print(f"checlp:{self.check_process}")
                action_list = self.possible_actions()
                
                if any(t[0] == 2 for t in action_list):
                    self.check_process = 0
                else:
                    self.check_process += 1

                if len(action_list) and self.check_process<=50:
                    self.update_action(action_list,action='initialize')
                else:
                    # for i,a in enumerate(action_list):
                    #     print(f"finale-{i}:{a}")
                    # print(f"t-{self.time_func.period()}")
                    
                    print("Scheduling ends")
                    break

    def update_action(self,action_list,action = False,present_value = 0):
        period = self.time_func.period()
    
        if period not in self.actions_taken:
            self.actions_taken[period] = {}

        if action:
            if 'initial' in self.actions_taken[period]:
                self.actions_taken[period]['final'] = action_list
            else:
                self.actions_taken[period]['initial'] = action_list
        else:
            if 'actions_taken' not in self.actions_taken[period]:
                self.actions_taken[period]['actions_taken'] = []
            if 'npv' not in self.actions_taken[period]:
                self.actions_taken[period]['npv'] = []
            self.add_action(self.actions_taken[period]['actions_taken'], self.actions_taken[period]['npv'], action_list,present_value)

    
    def add_action(self, action_list, npv_list, action,present_value):
        choice, block, amount = action
        # print(action_list)
        for i, (x, y, z) in enumerate(action_list):
            if x == choice and y == block:
                action_list[i] = (x, y, z + amount)
                break
        else:
            action_list.append(action)
        
        npv_list.append(present_value)
        # action_list.sort(key=lambda t: (t[0], t[1]))
        return#action_list

    def show_action(self):
        combined = {}
        max_len = 0
        for key, val in self.actions_taken.items():
            max_len = max(max_len, len(val['initial']), len(val['actions_taken']),len(val['npv']),2)
        for key, val in self.actions_taken.items():

            initial_list = val['initial'] + ['_']*(max_len - len(val['initial']))
            actions_list = val['actions_taken'] + ['_']*(max_len - len(val['actions_taken']))
            mine_high_limit = self.mine.Mining_capacity_upper[key]
            mine_low_limit =  self.mine.Mining_capacity_lower[key]
            process_high_limit = self.mine.Processing_capacity_upper[key]
            process_low_limit = self.mine.Processing_capacity_lower[key]
            capacity_list = [(mine_high_limit,mine_low_limit)] + [(process_high_limit,process_low_limit)] + ['_']*(max_len - 2)
            NPV_list = val['npv'] + ['_']*(max_len - len(val['npv']))
            # Store in dictionary
            combined[f"{key}_initial_value"] = initial_list
            combined[f"{key}_actions_taken"] = actions_list
            combined[f"{key}_NPV"] = NPV_list
            combined[f"{key}_capacities"] = capacity_list

        df = pd.DataFrame(combined)
        cols = df.columns.tolist()
        for i in range(0, len(cols), 4):
            # print(f"\nColumns {i} to {min(i+2, len(cols)-1)}:")
            print(df.iloc[:, i:i+4])
            print()
        return 
    
    def show_portions(self, choice = 1):
        if choice == 1:
            print("Extraction Portions:")
            data = self.extraction_portions.copy()
        else:
            print("Processing Portions:")
            data = self.processing_portions.copy()

        max_len = max(len(v) for v in data.values())

        padded = {f"Period_{k}":  [round(x * 100, 2) for x in v] + ['__']*(max_len - len(v)) for k, v in data.items()}

        df = pd.DataFrame(padded)
        return df
    
    def extraction_list(self,num_periods=1):
        model,y,x,b,ore_ton,block_ton,Capacity_low,l,Capacity_upper,Quantity_low,Quantity_upper = self.mine.run_deterministic_milp(number_of_periods=num_periods)
        result_df = pd.DataFrame()
        if model.status == GRB.OPTIMAL:
            optimal_npv = f"{model.objVal:.2f}"
            results = []
            
            for n in range(1,len(block_ton)):
                # print(f"\nBlock {n} schedule:")
                for t in range(1,num_periods+1):
                    y_val = y[n, t].x
                    x_val = x[n, t].x
                    b_val = b[n, t].x
                    mcl = Capacity_low[t] * l[t].x
                    mcu = Capacity_upper[t]
                    pcl = Quantity_low[t]
                    pcu = Quantity_upper[t]
                    bw= block_ton[n]
                    ow = ore_ton[n]
                    # Append to list
                    results.append({"block": n, "period": t, "y": y_val, "x": x_val,"BW": bw,"OW": ow,"b": b_val ,"MCU": mcu, "MCL": mcl, "PCU": pcu, "PCL": pcl,"block_extracted": y_val*bw,"block_processed": x_val*ow,"NPV":optimal_npv})
            result_df = pd.DataFrame(results)
            print("\nExtraction Schedule:")
            for row in result_df[result_df['y']>0].itertuples():
                    print(f"Block {row.block} mined in period {row.period} = Extraction {row.y*100:.2f}%, Extracted {row.block_extracted}, Processing {row.x*100:.2f}%")
        else:
            print("No possible solution.")
        return result_df
    
    def objective_func_npv(self,num_periods=1):
        model,y,x,b,ore_ton,block_ton,Capacity_low,l,Capacity_upper,Quantity_low,Quantity_upper = self.mine.run_deterministic_milp(number_of_periods=num_periods)
        # print(f"{model}")
        if model.status == GRB.OPTIMAL:
            optimal_npv = f"{model.objVal:.2f}"
            return f"({num_periods}, {optimal_npv})"
        else:
            print("No possible solution.")

    def action_choice(self,action_list,choice='random'):
        time_period = self.time_func.period()
        if choice == 'maximum':
            selected_action = max(action_list, key=lambda x: x[2])
            choice, block, max_amount,pv = selected_action
        elif choice =='minimum':
            selected_action = min(action_list, key=lambda x: x[2])
            choice, block, max_amount,pv = selected_action
        elif choice =='greedy':
            selected_action = max(action_list, key=lambda x: x[3])
            choice, block, amount,pv = selected_action
        else:
            selected_action = random.choice(action_list)
            choice, block, max_amount,pv = selected_action
            if choice == 1:
                capacity_max = self.mine.Mining_capacity_upper[time_period]
                capacity_min = self.mine.Mining_capacity_lower[time_period]
                remaining = capacity_max - self.mine.Mining_capacity_used[time_period]
            else:
                capacity_max = self.mine.Processing_capacity_upper[time_period]
                capacity_min = self.mine.Processing_capacity_lower[time_period]
                remaining = capacity_max - self.mine.Processing_capacity_used[time_period]

            amount = random.randint(capacity_min,min(max_amount,remaining)) if capacity_min < min(max_amount,remaining) else min(capacity_min,max_amount,remaining)
        
        pv = self.mine.calculate_NPV((choice, block, amount),time_period, c=True)
        pv = self.mine.update_npv(pv)
        return (choice, block, amount), pv
    
    def choosen_action(self,action):
        choice, block, max_amount = action
        time_period = self.time_func.period()
        if choice == 1:
            capacity_max = self.mine.Mining_capacity_upper[time_period]
            capacity_min = self.mine.Mining_capacity_lower[time_period]
            remaining = capacity_max - self.mine.Mining_capacity_used[time_period]
        else:
            capacity_max = self.mine.Processing_capacity_upper[time_period]
            capacity_min = self.mine.Processing_capacity_lower[time_period]
            remaining = capacity_max - self.mine.Processing_capacity_used[time_period]

        # amount = random.randint(capacity_min,min(max_amount,remaining)) if capacity_min < min(max_amount,remaining) else min(capacity_min,max_amount,remaining)
        pv = self.mine.calculate_NPV((choice, block, max_amount),time_period, c=True)
        pv = self.mine.update_npv(pv)
        return (choice, block, max_amount), pv
    
    def action(self, action_statement):
        choice, block_number, amount = action_statement

        # Normalize choice
        if choice in (1, '1', 'extraction', 'extract', 'ex'):
            choice_label = 'Extraction'
            xp = 'volume'
            verb = 'Extracting'
        elif choice in (2, '2', 'processing', 'process', 'proc', 'p'):
            choice_label = 'Processing'
            xp = 'concentration'
            verb = 'Processing'
        else:
            print("Wrong choice input")
            return
        
        t = self.time_func.period()
        # print(f"Executing action: {verb} {amount} from block {block_number} in Period {t}")

        # Validate block id exists
        if not isinstance(block_number, int) or block_number not in self.mine.block_properties:
            # print(f"Invalid block number: {block_number}")
            return
        block_id = block_number

        # Determine available tonnage and capacity bounds for current period
        
        if choice_label == 'Extraction':
            available = self.mine.block_tonnage[block_id]
            lower = self.mine.Mining_capacity_lower[t] if t < len(self.mine.Mining_capacity_lower) else None
            upper = self.mine.Mining_capacity_upper[t] if t < len(self.mine.Mining_capacity_upper) else None
        else:
            available = self.mine.ore_tonnage[block_id]
            lower = self.mine.Processing_capacity_lower[t] if t < len(self.mine.Processing_capacity_lower) else None
            upper = self.mine.Processing_capacity_upper[t] if t < len(self.mine.Processing_capacity_upper) else None

        # Basic checks (use "and" / combined conditions where appropriate)
        if amount > available:
            return
        if (upper is not None) and (amount > upper):
            return

        # All checks passed: perform update (pass current period)
        self.mine.update((xp, block_id, amount), time_func=self.time_func)
        return 1
        
    def possible_actions(self):
        action_list = self.mine.action_list(self.time_func.period())
        return action_list

    def time_tick(self):
        time_period = self.time_func.period()

        self.extraction_portions.setdefault(time_period, [])
        self.processing_portions.setdefault(time_period, [])
        # print(self.actions_taken[time_period])
        for actions in self.actions_taken[time_period]['actions_taken']:
            choice, block_id , amount_executed = actions
            if choice == 1:
                portion = amount_executed/self.mine.block_tonnage[block_id]
                self.extraction_portions[time_period].append(portion)
            else:
                if self.mine.ore_tonnage[block_id] > 0:
                    portion = amount_executed/self.mine.ore_tonnage[block_id]
                else:
                    portion = 0
                self.processing_portions[time_period].append(portion)

        self.time_func.tick()
        
    # def calculate_NPV(self,revenue=10,cost=5,discount_rate=0.1):
    #     vt = self.mine.revenue
    #     q = self.mine.cost
    #     r = self.mine.discount_rate
    #     total_revenue = 0
    #     total_cost = 0
    #     # print(cost)
    #     print(len(q))
    #     total_cost = sum(
    #         v * q[k] / (1 + r) ** k
    #         for k, vals in self.extraction_portions.items()
    #         for v in vals
    #     )
    #     # for k, vals in self.extraction_portions.items():
    #         # print(f"Period {k} Extraction Portions: {[round(v*100,2) for v in vals]}")
    #         # print(f"Period {k} Extraction Cost: {[round(v * q / (1 + r) ** k,2) for v in vals]}")
    #     total_revenue = sum(
    #         v * vt[k] / (1 + r) ** k
    #         for k, vals in self.processing_portions.items()
    #         for v in vals
    #     )

    #     npv = total_revenue - total_cost

    #     return npv

