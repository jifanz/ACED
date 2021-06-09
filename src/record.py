import pickle
import time


def load_record(filename):
    with open("./log/%s" % filename, 'rb') as file:
        record = pickle.load(file)
    return record


class Recorder:
    def __init__(self, data_name, model_name, func_name, idx):
        time_str = time.strftime("%Y%m%d-%H-%M-%S")
        self.name = "./log/%s_%s_%s_%s" % (data_name, model_name, func_name, idx)
        self._record = {}
        self._record['timestamp'] = time_str
        self.record = self._record
        self.previous = []
        
    def set_level(self, key=None):
        self.previous.append(self.record)
        if key is None:
            self.record = self._record
        else:
            if key in self.record:
                self.record = self.record[key]
            else:
                self.record[key] = {}
                self.record = self.record[key]
                
    def pop(self):
        if len(self.previous) > 0:
            self.record = self.previous.pop()
        else:
            self.record =self._record
            
    def record_var(self, key, val):
        self.record[key] = val
        
    def record_vars(self, keys, vals):
        assert len(keys)==len(vals)
        for k,v in zip(keys, vals):
            self.record[k] = v
    
    def append_var(self, key, val):
        if key in self.record:
            self.record[key].append(val)
        else:
            self.record[key] = [val]

    def append_vars(self, keys, vals):
        assert len(keys) == len(vals)
        for k,v in zip (keys, vals):
            self.append_var(k,v)
    
    def save(self):
        with open("%s.pkl" % self.name, 'wb') as file:
            pickle.dump(self._record, file, protocol=pickle.HIGHEST_PROTOCOL)
