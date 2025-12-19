class Time:
    def __init__(self):
        self.time_period = 1
    
    def tick(self):
        self.time_period += 1
    
    def period(self):
        return self.time_period