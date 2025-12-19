# num_periods = 365
Mine_properties = {
    "discount_rate": 0.1,
    "block_tonnage": 50,
    # "ore_tonnage": [60, 50, 40],
    "revenue": 100,
    "cost" : 10,
    "Mining_capacity_lower" : 20,
    "Mining_capacity_upper" : 70,
    "Processing_capacity_lower" : 0, 
    "Processing_capacity_upper" : 40,
    "Head_grade_lower" : [0.4],
    "Head_grade_upper": [6.0],
    "Head_grade" : [4, 0.6, 2],
    "constraint_control": [1,2,3,4,5,6],
    "progress" : []
}
concentration_func = []
if Mine_properties.get("ore_tonnage"):
    ore_tonnage = Mine_properties.get("ore_tonnage")
    if Mine_properties.get("block_tonnage"):
        block_ton = Mine_properties.get("block_tonnage")
    else:
        block_ton = 100
    concentration_func = [v/block_ton for v in ore_tonnage]
else:
    concentration_func = None