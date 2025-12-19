import plotly.graph_objects as go
from gurobipy import Model, GRB
import gurobipy as gp

class Mine:
    def __init__(self, arr, block_properties, HnD, num_periods, mine_properties):
        self.arr = arr
        self.block_properties = block_properties
        self.HnD = HnD
        self.num_x = arr.shape[2]
        self.num_y = arr.shape[1]
        self.num_z = arr.shape[0]
        self.num_periods = num_periods
        self.discount_rate = mine_properties.get("discount_rate", 0.1)
        
        self.block_tonnage = mine_properties.get("block_tonnage", None)
        if mine_properties.get("block_tonnage", None) is not None and isinstance(self.block_tonnage, (int, float)):
            self.block_tonnage = [self.block_tonnage] * (self.num_x * self.num_y * self.num_z)
        if self.block_tonnage is None:
            self.block_tonnage = [v["volume"] * 100  for v in block_properties.values()]
        
       
        if mine_properties.get("ore_tonnage", None) is not None and len(mine_properties.get("ore_tonnage", None)) != (self.num_x * self.num_y * self.num_z):
            self.ore_tonnage = mine_properties.get("ore_tonnage", None) * (self.num_x * self.num_y * self.num_z)
            self.ore_tonnage = self.ore_tonnage[:(self.num_x * self.num_y * self.num_z)]
        else:
            self.ore_tonnage = mine_properties.get("ore_tonnage", None)
        if self.ore_tonnage is None:
            self.ore_tonnage = [int(v["concentration"] * t) for v, t in zip(block_properties.values(), self.block_tonnage)]

        self.block_tonnage = [0] + self.block_tonnage
        self.ore_tonnage = [0] + self.ore_tonnage
        
        self.revenue = [mine_properties.get("revenue", 40) * ton for ton in self.ore_tonnage]
        self.cost = [mine_properties.get("cost", 3) * ton for ton in self.block_tonnage]

        self.Mining_capacity_lower =  [0] + [mine_properties.get("Mining_capacity_lower", None)] * (num_periods)
        self.Mining_capacity_upper = [0] + [mine_properties.get("Mining_capacity_upper", None)] * (num_periods)
        self.Mining_capacity_used = [0] * len(self.Mining_capacity_lower)

        self.Processing_capacity_lower = [0] + [mine_properties.get("Processing_capacity_lower", None)] * (num_periods)
        self.Processing_capacity_upper = [0] + [mine_properties.get("Processing_capacity_upper", None)] * (num_periods)
        self.Processing_capacity_used = [0] * len(self.Processing_capacity_lower)

        self.Head_grade_lower = [0] + mine_properties.get("Head_grade_lower", None) * (num_periods)
        self.Head_grade_upper = [0] + mine_properties.get("Head_grade_upper", None) * (num_periods)

        if mine_properties.get("Head_grade", None) is not None and len(mine_properties.get("Head_grade", None)) and isinstance(mine_properties.get("Head_grade", None), (int, float)):
            self.Head_grade = [mine_properties.get("Head_grade", None)] * (self.num_x * self.num_y * self.num_z + 1)
        elif mine_properties.get("Head_grade", None) is not None and len(mine_properties.get("Head_grade", None)) != (self.num_x * self.num_y * self.num_z):
            self.Head_grade = mine_properties.get("Head_grade", None) * (self.num_x * self.num_y * self.num_z)
            self.Head_grade = self.Head_grade[:(self.num_x * self.num_y * self.num_z+1)]
        else:
            self.Head_grade = mine_properties.get("Head_grade", None)
        self.Head_grade = [0] + self.Head_grade
        
        self.constraint_control = mine_properties.get("constraint_control", [0])
        self.mine_progress = mine_properties.get("progress")
        self.mine_extraction = [0.0]*len(self.block_tonnage)
        self.mine_process = [0.0]*len(self.block_tonnage)
        self.To_be_extracted = self.block_tonnage.copy() 
        self.To_be_processes = self.ore_tonnage.copy() 
        self.npv = 0

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
     
    def action_list(self,time_period = 1):
        done_blocks = [block for period in self.mine_progress for block in period]
        
        available_blocks = [
            k for k, deps in self.HnD.items()
            if not deps or all(d in done_blocks for d in deps)
        ]
        available_blocks_for_extract = [
            block for block in available_blocks
            if self.mine_extraction[block] != 1.0
        ]
        available_blocks_for_process = [
            block for block in available_blocks
            if self.mine_process[block] != 1.0 and self.mine_extraction[block] == 1.0
        ]
        # block_list = [k for k, v in self.HnD.items() if v == self.mine_progress]
        actions = []
        if self.num_periods < time_period:
            return actions
        if self.Mining_capacity_used[time_period] < self.Mining_capacity_upper[time_period]:
            for block in available_blocks_for_extract:
                possible_extraction_amount = self.Mining_capacity_upper[time_period] - self.Mining_capacity_used[time_period]
                possible_extraction_amount = min(possible_extraction_amount, self.To_be_extracted[block])
                if possible_extraction_amount < 0:
                    possible_extraction_amount = self.To_be_extracted[block]
                if possible_extraction_amount == 0:
                    continue
                if possible_extraction_amount > self.Mining_capacity_upper[time_period]:
                    possible_extraction_amount = self.Mining_capacity_upper[time_period]
                action = (1,block,possible_extraction_amount)
                value_per_unit = self.calculate_NPV(action,time_period)
                value_per_unit = round(self.calculate_NPV(action,time_period) / possible_extraction_amount,2) if possible_extraction_amount>0 else 0
                actions.append((1,block,possible_extraction_amount,value_per_unit))
        if self.Processing_capacity_used[time_period] < self.Processing_capacity_upper[time_period]:
            for block in available_blocks_for_process:
                possible_processing_amount = self.Processing_capacity_upper[time_period] - self.Processing_capacity_used[time_period]
                possible_processing_amount = min(possible_processing_amount,self.To_be_processes[block])
                if possible_processing_amount == 0:
                    continue
                if possible_processing_amount < 0:
                    possible_processing_amount = self.To_be_processes[block]
                if possible_processing_amount > self.Processing_capacity_upper[time_period]:
                    possible_processing_amount = self.Processing_capacity_upper[time_period]
                action = (2,block,possible_processing_amount)
                value_per_unit =  self.calculate_NPV(action,time_period)
                value_per_unit =  round(self.calculate_NPV(action,time_period) / possible_processing_amount,2) if possible_processing_amount>0 else 0
                if possible_processing_amount > 0:
                    actions.append((2,block,possible_processing_amount,value_per_unit))
        return actions
    
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

    def plot_block_model(self):
        voxel_properties = self.voxel_properties
        filled_x, filled_y, filled_z = [], [], []
        filled_i, filled_j, filled_k = [], [], []
        filled_colors = []
        filled_vertex_count = 0

        label_x, label_y, label_z, label_text = [], [], [], []

        for voxel_id, props in voxel_properties.items():
            i, j, k = props["indices"]
            conc = props["concentration"]

            # Define cube corners
            x0, y0, z0 = i, j, k
            x1, y1, z1 = i + 1, j + 1, k + 1
            verts = [
                (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)
            ]
            faces = [
                (0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7),
                (0, 4, 5), (0, 5, 1), (2, 6, 7), (2, 7, 3),
                (1, 5, 6), (1, 6, 2), (0, 3, 7), (0, 7, 4)
            ]

            # Add cube vertices and faces
            filled_x.extend([v[0] for v in verts])
            filled_y.extend([v[1] for v in verts])
            filled_z.extend([v[2] for v in verts])
            filled_colors.extend([conc] * 8)

            for f in faces:
                filled_i.append(f[0] + filled_vertex_count)
                filled_j.append(f[1] + filled_vertex_count)
                filled_k.append(f[2] + filled_vertex_count)
            filled_vertex_count += 8

            # Add label text
            cx, cy, cz = props["coords"]
            label_x.append(cx)
            label_y.append(cy)
            label_z.append(cz)
            label_text.append(f"{voxel_id}")

        # Step 4: Build the 3D plot
        data_traces = []
        data_traces.append(go.Mesh3d(
            x=filled_x, y=filled_y, z=filled_z,
            i=filled_i, j=filled_j, k=filled_k,
            vertexcolor=filled_colors,
            intensity=filled_colors,
            colorscale="dense",
            opacity=0.8,
            cmin=0, cmax=1,
            colorbar=dict(title="Ore Concentration", x=1.15, len=0.5, y=0.75),
            name="Blocks",
            showlegend=True
        ))

        # data_traces.append(go.Scatter3d(
        #     x=label_x, y=label_y, z=label_z,
        #     text=label_text, mode="text",
        #     textfont=dict(color="white", size=12),
        #     name="Labels",
        #     showlegend=False
        # ))

        # Camera setup to view the front face (x=0 plane)
        camera = dict(
            eye=dict(x=-2.5, y=0, z=0), 
            up=dict(x=0, y=1, z=0),       
            center=dict(x=0, y=0, z=0)
        )

        layout = go.Layout(
            title="Block Model Visualization (Front View)",
            scene=dict(
                xaxis_title="X", 
                yaxis_title="Y", 
                zaxis_title="Z",
                aspectmode='cube',
                xaxis=dict(range=[0, max(filled_x)]),
                yaxis=dict(autorange='reversed'),
                zaxis=dict(autorange=True)
            ),
            scene_camera=camera,
            margin=dict(l=0, r=0, b=0, t=50),
            paper_bgcolor='#1a202c',
            plot_bgcolor='#1a202c',
            font=dict(color='white')
        )

        fig = go.Figure(data=data_traces, layout=layout)
        fig.show()
    def plot_block_model_indices(self):
        voxel_properties = self.voxel_properties
        voxel_properties = self.voxel_properties
        filled_x, filled_y, filled_z = [], [], []
        filled_i, filled_j, filled_k = [], [], []
        filled_colors = []
        filled_vertex_count = 0

        label_x, label_y, label_z, label_text = [], [], [], []
        voxel_traces = []
        y_indices = []

        # ---- Build each voxel as its own Mesh3d trace ----
        for voxel_id, props in voxel_properties.items():
            i, j, k = props["indices"]
            conc = props["concentration"]

            x0, y0, z0 = i, j, k
            x1, y1, z1 = i + 1, j + 1, k + 1

            verts = [
                (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)
            ]
            faces = [
                (0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7),
                (0, 4, 5), (0, 5, 1), (2, 6, 7), (2, 7, 3),
                (1, 5, 6), (1, 6, 2), (0, 3, 7), (0, 7, 4)
            ]
            filled_x.extend([v[0] for v in verts])
            filled_y.extend([v[1] for v in verts])
            filled_z.extend([v[2] for v in verts])
            filled_colors.extend([conc] * 8)

            for f in faces:
                filled_i.append(f[0] + filled_vertex_count)
                filled_j.append(f[1] + filled_vertex_count)
                filled_k.append(f[2] + filled_vertex_count)
            filled_vertex_count += 8

            x, y, z = zip(*verts)
            i_idx = [f[0] for f in faces]
            j_idx = [f[1] for f in faces]
            k_idx = [f[2] for f in faces]

            trace = go.Mesh3d(
                x=x, y=y, z=z,
                i=i_idx, j=j_idx, k=k_idx,
                vertexcolor=[conc]*8,
                intensity=[conc]*8,
                colorscale="dense",
                opacity=0.85,
                cmin=0, cmax=1,
                showscale=True,
                name=f"Voxel {voxel_id}",
                visible=True
            )

            voxel_traces.append(trace)
            y_indices.append(j)

        max_y = max(y_indices)

        # ---- Build SLIDER steps ----
        steps = []
        for ymax in range(max_y + 1):
            visible = [(y_indices[n] <= ymax) for n in range(len(y_indices))]
            step = dict(
                method="update",
                args=[{"visible": visible},
                    {}]
            )
            steps.append(step)

        middle_step = len(steps) // 2  - 1

        sliders = [dict(
            active=middle_step,
            pad={"t": 20},
            steps=steps
        )]
        # data_traces = []
        # data_traces.append(go.Mesh3d(
        #     x=filled_x, y=filled_y, z=filled_z,
        #     i=filled_i, j=filled_j, k=filled_k,
        #     vertexcolor=filled_colors,
        #     intensity=filled_colors,
        #     colorscale="dense",
        #     opacity=0.8,
        #     cmin=0, cmax=1,
        #     colorbar=dict(title="Ore Concentration", x=1.15, len=0.5, y=0.75),
        #     name="Blocks",
        #     showlegend=True
        # ))

        # ---- Layout ----
        layout = go.Layout(
            title="Block Model Slice Viewer (Y-axis Slab)",
            sliders=sliders,
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="cube",
                xaxis=dict(autorange='reversed'),
                yaxis=dict(range=[0, max_y + 1]),
                zaxis=dict(autorange=True),
                camera=dict(
                    eye=dict(x=0, y=2.5, z=0),  # Look along +Y axis
                    up=dict(x=1, y=0, z=0)      # Make X the vertical screen axis
                )
            ),
            paper_bgcolor="#1a202c",
            plot_bgcolor="#1a202c",
            font=dict(color="white")
        )


        fig = go.Figure(data=voxel_traces, layout=layout)
        fig.show()
    
    def plot_block_model_indice(self):
        voxel_properties = self.voxel_properties
        voxel_properties = self.voxel_properties
        filled_x, filled_y, filled_z = [], [], []
        filled_i, filled_j, filled_k = [], [], []
        filled_colors = []
        filled_vertex_count = 0

        label_x, label_y, label_z, label_text = [], [], [], []
        voxel_traces = []
        y_indices = []

        # ---- Build each voxel as its own Mesh3d trace ----
        for voxel_id, props in voxel_properties.items():
            i, j, k = props["indices"]
            conc = props["concentration"]

            max_x = max([props["indices"][0] for props in voxel_properties.values()])
            max_y = max([props["indices"][1] for props in voxel_properties.values()])
            max_z = max([props["indices"][2] for props in voxel_properties.values()])

            # Create unit cube: i to i+1, j to j+1, k to k+1
            x0, y0, z0 = i+2, j, k
            x1, y1, z1 = i +3, j + 1, k + 1

            verts = [
                (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)
            ]
            faces = [
                (0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7),
                (0, 4, 5), (0, 5, 1), (2, 6, 7), (2, 7, 3),
                (1, 5, 6), (1, 6, 2), (0, 3, 7), (0, 7, 4)
            ]
            filled_x.extend([v[0] for v in verts])
            filled_y.extend([v[1] for v in verts])
            filled_z.extend([v[2] for v in verts])
            filled_colors.extend([conc] * 8)

            for f in faces:
                filled_i.append(f[0] + filled_vertex_count)
                filled_j.append(f[1] + filled_vertex_count)
                filled_k.append(f[2] + filled_vertex_count)
            filled_vertex_count += 8

            x, y, z = zip(*verts)
            i_idx = [f[0] for f in faces]
            j_idx = [f[1] for f in faces]
            k_idx = [f[2] for f in faces]

            trace = go.Mesh3d(
                x=x, y=y, z=z,
                i=i_idx, j=j_idx, k=k_idx,
                vertexcolor=[conc]*8,
                intensity=[conc]*8,
                colorscale="dense",
                opacity=1.0,
                cmin=0, cmax=1,
                colorbar=dict(title="Ore\nConc."
                # , x=1.15, len=0.5, y=0.75
                ),
                showscale=True,
                flatshading=True,
                lighting=dict(ambient=1, diffuse=0, specular=0, roughness=1, fresnel=0),
                name=f"Voxel {voxel_id}",
                visible=True
            )
            
            voxel_traces.append(trace)
            y_indices.append(j)

        max_y = max(y_indices)

        # ---- Build SLIDER steps ----
        steps = []
        for ymax in range(max_y + 1):
            visible = [(y_indices[n] <= ymax) for n in range(len(y_indices))]
            step = dict(
                method="update",
                args=[{"visible": visible},
                    {}]
            )
            steps.append(step)

        middle_step = len(steps) // 2  - 1

        sliders = [dict(
            active=middle_step,
            pad={"t": 20},
            steps=steps
        )]
        # data_traces = []
        # data_traces.append(go.Mesh3d(
        #     x=filled_x, y=filled_y, z=filled_z,
        #     i=filled_i, j=filled_j, k=filled_k,
        #     vertexcolor=filled_colors,
        #     intensity=filled_colors,
        #     colorscale="dense",
        #     opacity=0.8,
        #     cmin=0, cmax=1,
        #     colorbar=dict(title="Ore Concentration", x=1.15, len=0.5, y=0.75),
        #     name="Blocks",
        #     showlegend=True
        # ))

        # ---- Layout ----
        layout = go.Layout(
            title="Mine Model Plot Cross Section Viewer",
            sliders=sliders,
            scene=dict(
                xaxis=dict(
                    title="Mine Depth",
                    showgrid=True,
                    gridcolor="lightgray",
                    gridwidth=3,
                    range=[max_x + 3, 0],
                    showbackground=True,
                    backgroundcolor="rgba(230,230,230,0.5)",
                    zeroline=True,
                    showticklabels=False
                ),
                yaxis=dict(
                    title="Mine Width",
                    showgrid=True,
                    gridcolor="lightgray",
                    gridwidth=3,
                    range=[0, max_y + 2],
                    showbackground=True,
                    backgroundcolor="rgba(230,230,230,0.5)",
                    zeroline=True,
                    showticklabels=False
                ),
                zaxis=dict(
                    title="Mine Length",
                    showgrid=True,
                    gridcolor="lightgray",
                    gridwidth=3,
                    range=[0, max_z + 2],
                    showbackground=True,
                    backgroundcolor="rgba(200,200,200,0.3)",
                    zeroline=True,
                    showticklabels=False
                ),
                aspectmode="cube",
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=1.0),  # Closer camera
                    up=dict(x=1, y=0, z=0),
                    center=dict(x=0, y=0, z=0)
                )
            ),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="black"),
            margin=dict(l=0, r=0, t=40, b=0)  # Reduce margins
        )

        fig = go.Figure(data=voxel_traces, layout=layout)
        fig.show()


    def plot_block_model_indice_highlight(self,highlight_blocks=None,period_to_print=None):
        voxel_properties = self.voxel_properties
        voxel_properties = self.voxel_properties
        filled_x, filled_y, filled_z = [], [], []
        filled_i, filled_j, filled_k = [], [], []
        filled_colors = []
        filled_vertex_count = 0

        highlight_x, highlight_y, highlight_z = [], [], []
        highlight_i, highlight_j, highlight_k = [], [], []
        highlight_colors = []
        highlight_vertex_count = 0

        label_x, label_y, label_z, label_text = [], [], [], []
        voxel_traces = []
        y_indices = []

        highlight_dict = {}
        if highlight_blocks:
            for item in highlight_blocks:
                if isinstance(item, tuple):
                    voxel_id, conc = item
                    highlight_dict[voxel_id] = conc
                else:
                    # If just an ID, use original concentration
                    highlight_dict[item] = None

        # ---- Build each voxel as its own Mesh3d trace ----
        for voxel_id, props in voxel_properties.items():
            i, j, k = props["indices"]
            conc = props["concentration"]

            max_x = max([props["indices"][0] for props in voxel_properties.values()])
            max_y = max([props["indices"][1] for props in voxel_properties.values()])
            max_z = max([props["indices"][2] for props in voxel_properties.values()])

            # Create unit cube: i to i+1, j to j+1, k to k+1
            x0, y0, z0 = i+2, j, k
            x1, y1, z1 = i +3, j + 1, k + 1

            verts = [
                (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)
            ]
            faces = [
                (0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7),
                (0, 4, 5), (0, 5, 1), (2, 6, 7), (2, 7, 3),
                (1, 5, 6), (1, 6, 2), (0, 3, 7), (0, 7, 4)
            ]
            
            # Check if this block should be highlighted
            is_highlighted = voxel_id in highlight_dict
            
            # Use custom concentration if provided, otherwise use original
            display_conc = highlight_dict.get(voxel_id, conc) if is_highlighted else conc
            if display_conc is None:
                display_conc = conc
            
            if is_highlighted:
                # Add to highlighted blocks
                highlight_x.extend([v[0] for v in verts])
                highlight_y.extend([v[1] for v in verts])
                highlight_z.extend([v[2] for v in verts])
                highlight_colors.extend([display_conc]*8)
                
                for f in faces:
                    highlight_i.append(f[0] + highlight_vertex_count)
                    highlight_j.append(f[1] + highlight_vertex_count)
                    highlight_k.append(f[2] + highlight_vertex_count)
                highlight_vertex_count += 8
            else:
                # Add to regular blocks
                filled_x.extend([v[0] for v in verts])
                filled_y.extend([v[1] for v in verts])
                filled_z.extend([v[2] for v in verts])
                filled_colors.extend([conc]*8)
                
                for f in faces:
                    filled_i.append(f[0] + filled_vertex_count)
                    filled_j.append(f[1] + filled_vertex_count)
                    filled_k.append(f[2] + filled_vertex_count)
                filled_vertex_count += 8

            # Add label at block center
            cx, cy, cz = props["coords"]
            label_x.append(cx)
            label_y.append(cy)
            label_z.append(cz)
            label_text.append(f"{voxel_id}")

            x, y, z = zip(*verts)
            i_idx = [f[0] for f in faces]
            j_idx = [f[1] for f in faces]
            k_idx = [f[2] for f in faces]

            trace = go.Mesh3d(
                x=x, y=y, z=z,
                i=i_idx, j=j_idx, k=k_idx,
                vertexcolor=[conc]*8,
                intensity=[conc]*8,
                colorscale="dense",
                opacity=1.0,
                cmin=0, cmax=1,
                colorbar=dict(title="Ore\nConc."
                # , x=1.15, len=0.5, y=0.75
                ),
                showscale=True,
                flatshading=True,
                lighting=dict(ambient=1, diffuse=0, specular=0, roughness=1, fresnel=0),
                name=f"Voxel {voxel_id}",
                visible=True
            )
            # Add highlighted blocks (with higher opacity and different color)
            if highlight_x:
                highlight_trace = go.Mesh3d(
                    x=highlight_x, y=highlight_y, z=highlight_z,
                    i=highlight_i, j=highlight_j, k=highlight_k,
                    vertexcolor=highlight_colors,
                    intensity=highlight_colors,
                    colorscale="YlOrRd",  # Different colorscale for highlighted blocks
                    opacity=0.95,
                    cmin=0, cmax=1,
                    colorbar=dict(title="Extraction concentration", x=1.15, len=0.3, y=0.25),
                    name=f"Period: {period_to_print} ",
                    # {m_l}-{m_u}
                    showlegend=True
                )
            
            voxel_traces.append(trace)
            voxel_traces.append(highlight_trace)
            y_indices.append(j)

        max_y = max(y_indices)

        # ---- Build SLIDER steps ----
        steps = []
        for ymax in range(max_y + 1):
            visible = [(y_indices[n] <= ymax) for n in range(len(y_indices))]
            step = dict(
                method="update",
                args=[{"visible": visible},
                    {}]
            )
            steps.append(step)

        middle_step = len(steps) // 2  - 1

        sliders = [dict(
            active=middle_step,
            pad={"t": 20},
            steps=steps
        )]
        # data_traces = []
        # data_traces.append(go.Mesh3d(
        #     x=filled_x, y=filled_y, z=filled_z,
        #     i=filled_i, j=filled_j, k=filled_k,
        #     vertexcolor=filled_colors,
        #     intensity=filled_colors,
        #     colorscale="dense",
        #     opacity=0.8,
        #     cmin=0, cmax=1,
        #     colorbar=dict(title="Ore Concentration", x=1.15, len=0.5, y=0.75),
        #     name="Blocks",
        #     showlegend=True
        # ))

        # ---- Layout ----
        layout = go.Layout(
            title="Mine Model Plot Cross Section Viewer",
            sliders=sliders,
            scene=dict(
                xaxis=dict(
                    title="Mine Depth",
                    showgrid=True,
                    gridcolor="lightgray",
                    gridwidth=3,
                    range=[max_x + 3, 0],
                    showbackground=True,
                    backgroundcolor="rgba(230,230,230,0.5)",
                    zeroline=True,
                    showticklabels=False
                ),
                yaxis=dict(
                    title="Mine Width",
                    showgrid=True,
                    gridcolor="lightgray",
                    gridwidth=3,
                    range=[0, max_y + 2],
                    showbackground=True,
                    backgroundcolor="rgba(230,230,230,0.5)",
                    zeroline=True,
                    showticklabels=False
                ),
                zaxis=dict(
                    title="Mine Length",
                    showgrid=True,
                    gridcolor="lightgray",
                    gridwidth=3,
                    range=[0, max_z + 2],
                    showbackground=True,
                    backgroundcolor="rgba(200,200,200,0.3)",
                    zeroline=True,
                    showticklabels=False
                ),
                aspectmode="cube",
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=1.0),  # Closer camera
                    up=dict(x=1, y=0, z=0),
                    center=dict(x=0, y=0, z=0)
                )
            ),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="black"),
            margin=dict(l=0, r=0, t=40, b=0)  # Reduce margins
        )

        fig = go.Figure(data=voxel_traces, layout=layout)
        fig.show()
    def display_npv(self,time_period):
        print(f"Net Present value of time period {time_period} is: {self.npv}")

    def update_npv(self,pv):
        # print(f"Updating NPV: {self.npv} = {pv}")
        self.npv += round(pv,2)
        return self.npv
    
    def calculate_NPV(self,action,t,c = None):
        choice, block_id, amount = action
        vt = self.revenue[block_id]
        q = self.cost[block_id]
        r = self.discount_rate
        # print(f"Calculating NPV for action: {q} at time {t}")
        if choice == 1:
            cost = q*amount/self.block_tonnage[block_id]
            # return round(self.npv - (cost / ((1 + r) ** t)), 2)
            return -(cost / ((1 + r) ** t))
        else:
            if self.ore_tonnage[block_id]==0:
                return 0
            revenue = vt * amount/self.ore_tonnage[block_id]
            # return round(self.npv + (revenue / ((1 + r) ** t)),2)
            return (revenue / ((1 + r) ** t))

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
            model.setParam("TimeLimit", 18000)
                    
        # model.write("deterministic_model.lp") 
        model.optimize()
        # print("------------------------------------------------------\n")
            
        return model,y,x,b,o,tt,Cl,l,Cu,Ql,Qu    