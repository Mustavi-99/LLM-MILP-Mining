import numpy as np

class mining_data:
    def __init__(self, shape):
        self.shape = shape
        self.x, self.y, self.z = shape
        self.arr = np.arange(1, self.x*self.y*self.z + 1).reshape(self.z, self.y, self.x)
    
    def gaussian_2d(self,h, w, sigma, center):
        y, x = np.indices((h, w))
        cy, cx = center
        g = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2 + 1e-12))
        return g

    def drift_straight_stack(self,
            T, H, W,
            start_center,
            end_center=None,
            sigma_start=6.0,
            sigma_end=0.2,
            peak_start=0.75,
            peak_end=1.0,
            round_decimals=2,
            tiny_thresh=1e-3
        ):
            """Gaussian drifts in a straight line toward the center."""
            if end_center is None:
                end_center = (H/2, W/2)
            
            cy0, cx0 = start_center
            cy1, cx1 = end_center
            
            centers_y = np.linspace(cy0, cy1, T)
            centers_x = np.linspace(cx0, cx1, T)
            sigmas = np.linspace(sigma_start, sigma_end, T)
            peaks  = np.linspace(peak_start, peak_end, T)
            
            stack = np.zeros((T, H, W))
            
            for i in range(T):
                g = self.gaussian_2d(H, W, sigmas[i], (centers_y[i], centers_x[i]))
                g = (g / g.max()) * peaks[i]
                if round_decimals is not None:
                    g = np.round(g, round_decimals)
                g[g < tiny_thresh] = 0
                stack[i] = g
            
            return stack.flatten().tolist()
    
    def build(self,concentration_func=None):
        x, y, z = self.x, self.y, self.z
        arr = self.arr    
        HnD = {}
        block_properties = {}
        if concentration_func is not None:
            len_arr = self.x*self.y*self.z
            if len(concentration_func)< len_arr:
                concentration_func = concentration_func * len_arr
                concentration_func = [0] + concentration_func[:len_arr]
        else:
            stack_straight = self.drift_straight_stack(
                                    T=z, H=x, W=y,
                                    start_center=(x, int((y/2)-1)),
                                    end_center=None     # automatically go to real center
                                )
            func_val = [0] + stack_straight
        for b in range(z):
            for r in range(y):
                for c in range(x):
                    voxel_id = int(arr[b, r, c])
                    if concentration_func is None:
                        # if voxel_id < 26:
                        #     con = 0.2 + (0.6 - 0.2) * np.random.rand()
                        # elif voxel_id < 50:
                        #     con = 0.4 + (0.7 - 0.4) * np.random.rand()
                        # else:
                        #     con = (0.5 - 0.4) * np.random.rand()
                        con = func_val[voxel_id]
                    else:
                        con = concentration_func[voxel_id]  # or any custom logic

                    block_properties[voxel_id] = {
                        "indices": (b, r, c),
                        "volume": 1.0,
                        "concentration": float(con),
                        "coords": (b + 0.5, r + 0.5, c + 0.5)
                    }
        
        coords_dict = {k: v['coords'] for k, v in block_properties.items()}
        for k, v in coords_dict.items():
            c_1, c_2, c_3 = v

            if c_1 == 0.5 and c_2 == 0.5:
                HnD[k] = []
                continue

            keys = []
            targets = [
                (c_1-1,c_2-1,c_3-1),
                (c_1-1,c_2-1,c_3),
                (c_1-1,c_2-1,c_3+1),

                (c_1-1,c_2,c_3-1),
                (c_1-1,c_2,c_3),
                (c_1-1,c_2,c_3+1),

                
                (c_1-1,c_2+1,c_3-1),
                (c_1-1,c_2+1,c_3),
                (c_1-1,c_2+1,c_3+1),
                
            ]
            for target in targets:
                match = next((k2 for k2, v2 in coords_dict.items() if v2 == target), None)
                if match is not None:
                    keys.append(match)

            HnD[k] = keys

        return arr, block_properties, HnD
