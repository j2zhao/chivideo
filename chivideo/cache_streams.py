import json
from chivideo.streams import *

class CacheStream(DataStream):
    def __init__(self, name):
        super().__init__(name)
        self.cache = []
        self.min_frame = 0
        self.keep_all = False
        self.operator = None

    def set_op(self, operator):
        self.operator = operator

    def clear(self):
        self.cache = []
        self.min_frame = 0
        
        for iter in self.iters:
            self.iters[iter] = [-1, 0]

    #only used by GraphManager -> to return results 
    #if you need to keep some datastream, set the results parameter on GraphManager run 
    def set_keep_all(self, val = True):
        self.keep_all = val

    def _next(self, op_name):
        i = self.iters[op_name][0]
        
        index = i - self.min_frame
        dif = i - (self.min_frame + len(self.cache)) + 1
        
        while True:
            if dif > 0:
                ret = self.operator.next()
                if not ret:
                    self.iters[op_name] = None
                    return None
                dif = i - (self.min_frame + len(self.cache)) + 1
            else:
                break

        frame = self.cache[index]

        if not self.keep_all:
            min_next = float('inf')
            for key in self.iters:
                if self.iters[key][1] < min_next:
                    min_next = self.iters[key][1]

            if min_next > self.min_frame:
                k = min_next - self.min_frame
                self.cache = self.cache[k:]
                self.min_frame = min_next
        
        return frame
    
    
    def next(self, op_name, next_step = 1):
        super().next(op_name, next_step)
        return self._next(self, op_name)

    def skip_next(self, op_name, skip, next_step=1):
        super().skip_next(op_name, skip, next_step)
        return self._next(self, op_name)

    # should only be called by operator
    def insert(self, value):
        self.cache.append(value)

    def all(self):
        return self.cache

    @staticmethod
    def init_mat():
        return []
    
    @staticmethod
    def append(data, prev):
        return prev.append(data)
    
    @staticmethod
    def materialize(data, fp = None):
        if not fp:
            return json.dumps(data)
        else:
            return json.dump(data, fp)

class DictCacheStream(CacheStream):
    def __init__(self, name):
        super().__init__(name)
        self.cache = {}
        self.keep_all = False
        self.operator = None
        self.min_frame = -1

    def set_op(self, operator):
        self.operator = operator
        if not hasattr(operator, 'skip_next'):
            raise NotImplementedError()

    def clear(self):
        self.caches = {}
        self.min_frame = 0

    def _next(self, op_name):
        i = self.iters[op_name][0]
        
        if i not in self.cache:
            ret = self.operator.skip_next(self.name, i)
            if not ret:
                self.iters[op_name] = None
                return None

        frame = self.cache[i]

        if not self.keep_all:
            min_next = float('inf')
            for key in self.iters:
                if self.iters[key][1] < min_next:
                    min_next = self.iters[key][1]

            if min_next > self.min_frame:
                for key in self.cache:
                    if key < min_next:
                        del self.cache[key]
                    self.min_frame = min_next

        return frame
    
    # should only be called by operator
    def insert(self, key, value):
        self.cache[key] = value      

    @staticmethod
    def init_mat():
        return {}
    
    @staticmethod
    def append(data, prev):
        prev[data[0]] = data[1]
        return prev

class SampleCacheStream(CacheStream):
    ''' Every index value is multiplied by sample rate
    '''
    def __init__(self, name, sample_rate, skips = True):
        super().__init__(name)
        self.cache = []
        self.sample_rate = sample_rate
        self.min_frame = 0
        self.keep_all = False
        self.operator = None
        self.skips = skips

    def _next(self, op_name):
        i = self.iters[op_name][0]

        index = i - self.min_frame
        dif = i - (self.min_frame + len(self.cache)) + 1

        if dif > 0:
            if not hasattr(self.operator, 'skip_next') or not self.skips:
                while True:
                    if dif > 0:
                        ret = self.operator.next()
                        if not ret:
                            self.iters[op_name] = None
                            return None
                        dif = i - (self.min_frame + len(self.cache)) + 1
                    else:
                        break
            else:
                while True:
                    if dif > 0:
                        n = (self.min_frame + len(self.cache))*self.sample_rate
                        ret = self.operator.skip_next(self.name, n)
                        if not ret:
                            self.iters[op_name] = None
                            return None
                        dif = i - (self.min_frame + len(self.cache)) + 1

        frame = self.cache[index]

        if not self.keep_all:
            min_next = float('inf')
            for key in self.iters:
                if self.iters[key][1] < min_next:
                    min_next = self.iters[key][1]

            if min_next > self.min_frame:
                k = min_next - self.min_frame
                self.cache = self.cache[k:]
                self.min_frame = min_next
        
        return frame
    

    # should only be called by operator
    def insert(self, key, value):
        if key == (self.min_frame + len(self.cache))*self.sample_rate:
            self.cache.append(value)     
        elif key > (self.min_frame + len(self.cache))*self.sample_rate:
            raise ValueError()
        else:
            pass

class PassiveCacheStream(CacheStream):

    def _next(self, i):        
        index = i - self.min_frame
        dif = i - (self.min_frame + len(self.cache)) + 1
        if dif > 0:
            return None
        else:
            return self.cache[index]
    
    def _reduce_cache(self):
        min_next = float('inf')
        for key in self.iters:
            if self.iters[key][1] < min_next:
                min_next = self.iters[key][1]

        if min_next > self.min_frame:
            k = min_next - self.min_frame
            self.cache = self.cache[k:]
            self.min_frame = min_next
    
    
    def next(self, op_name, next_step = 1):
        if op_name not in self.iters:
            return None

        
        i = self.iters[op_name][1]
        frame = self._next(self, i)
        if frame == None:
            return None
        else:
            self.iters[op_name][0] = i
            self.iters[op_name][1] += next_step
            if not self.keep_all:
                self._reduce_cache()
            return frame


    def skip_next(self, op_name, skip, next_step=1):
        if op_name not in self.iters:
            return None
        
        i = self.iters[op_name][0] + skip
        frame = self._next(self, i)
        if frame == None:
            return None
        else:
            self.iters[op_name][0] = i
            self.iters[op_name][1] = i + next_step
            if not self.keep_all:
                self._reduce_cache()
            return frame
    
class PassiveDictCacheStream(DictCacheStream):

    def set_op(self, operator):
        self.operator = operator

    def _next(self, i):        
        if i in self.cache:
            return self.cache[i]
        else:
            return None
    
    def _reduce_cache(self):
        min_next = float('inf')
        for key in self.iters:
            if self.iters[key][1] < min_next:
                min_next = self.iters[key][1]

        if min_next > self.min_frame:
            for key in self.cache:
                if key < min_next:
                    del self.cache[key]
            self.min_frame = min_next
    
    
    def next(self, op_name, next_step = 1):
        if op_name not in self.iters:
            return None
        
        i = self.iters[op_name][1]
        frame = self._next(self, i)
        if frame == None:
            return None
        else:
            self.iters[op_name][0] = i
            self.iters[op_name][1] += next_step
            if not self.keep_all:
                self._reduce_cache()
            return frame


    def skip_next(self, op_name, skip, next_step=1):
        if op_name not in self.iters:
            return None
        
        i = self.iters[op_name][0] + skip
        frame = self._next(self, i)
        if frame == None:
            return None
        else:
            self.iters[op_name][0] = i
            self.iters[op_name][1] = i + next_step
            if not self.keep_all:
                self._reduce_cache()
            return frame