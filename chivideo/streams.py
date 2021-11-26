# TODO: Add error if next is called before initialize




import json
import os
import tempfile
from shutil import copyfile
from subprocess import Popen
from timeit import default_timer as timer
import numpy as np
import cv2
import threading

from deeplens.utils.error import *


class DataStream():
    def __init__(self, name):
        self.name = name
        self.iters = {}

    def add_iter(self, op_name):
        self.iters[op_name] = [-1 , 0]

    def update_iters(self, index):
        for op_name in self.iters:
            if self.iters[op_name][0] > index:
                raise ValueError()
            self.iters[op_name][0] = index - 1
            self.iters[op_name][1] = index

    def del_iter(self, op_name):
        del self.iters[op_name]

    def next(self, op_name, next_step = 1):
        if op_name not in self.iters:
            return None
        if self.iters[op_name] == None:
            return None
        i = self.iters[op_name][1]
        self.iters[op_name][0] = i
        self.iters[op_name][1] += next_step
    
    def skip_next(self, op_name, skip, next_step = 1):
        if op_name not in self.iters:
            return None
        if self.iters[op_name] == None:
            return None
        i = self.iters[op_name][0] + skip

        self.iters[op_name][0] = i
        self.iters[op_name][1] = i + next_step # document this behavior

    @staticmethod
    def init_mat():
        raise NotImplementedError("init_mat not implemented")

    @staticmethod
    def append(data, prev):
        raise NotImplementedError("append not implemented")

    @staticmethod
    def materialize(self, data):
        raise NotImplementedError("materialize not implemented")

class JSONListStream(DataStream):
    def __init__(self, data, name, limit = -1, is_file = False, is_list = False):
        super().__init__(name)
        self.data = []
        self.limit = limit
        if data is None:
            return 
        if is_file:
            if type(data) == str:
                    files = [data]
            else:
                files = data
            for file in files:
                with open(file, 'r') as f:
                    if is_list:
                        self.data = self.data + json.load(f)
                    else:
                        self.data = self.data.append(json.load(f))
        else:
            self.data = data

    def _next(self, op_name):
        i = self.iters[op_name][0]
        if i >= len(self.data) or (i > self.limit and self.limit > 0):
            self.iters[op_name] = None
            return None
        return self.data[i]

    def next(self, op_name, next_step = 1):
        super().next(op_name, next_step)
        return self._next(self, op_name)

    def skip_next(self, op_name, skip, next_step = 1):
        super().skip_next(op_name, skip, next_step)
        return self._next(self, op_name)

    
    def serialize(self, fp = None):
        if not fp:
            return json.dumps(self.data)
        else:
            return json.dump(self.data, fp)
    
    def size(self):
        return len(self.data)

    @staticmethod
    def init_mat():
        return []

    @staticmethod
    def append(data, prev):
        return prev.data.append(data)

    @staticmethod
    def materialize(data, fp = None):
        if not fp:
            return json.dumps(data)
        else:
            return json.dump(data, fp)

class JSONDictStream(DataStream):
    def __init__(self, data, name, limit = 0):
        super().__init__(name)
        self.data = {}
        if data is not None:
            if type(data) == str:
                files = [data]
            else:
                files = data
            for file in files:
                with open(file, 'r') as f:
                    self.data = self.data.update(json.load(f))
        self.limit = limit


    def _next(self, op_name):
        i = self.iters[op_name][0]
        if self.limit and i >= self.limit:
            self.iters[op_name] = None
            return None
        if i in self.data:
            return self.data[i]
        else:
            return None

    def next(self, op_name, next_step=1):
        super().next(op_name, next_step=next_step)
        return self._next(op_name)        
        
        
    def skip_next(self, op_name, skip, next_step = 1):
        super().skip_next(op_name, skip, next_step)

        return self._next(op_name)

    def size(self):
        return self.limit
    
    def update_limit(self, limit):
        self.limit = limit

    def add(self, data, index):
        self.data[index] = data

    def update(self, stream):
        self.data.update(stream.data)

    def serialize(self, fp = None):
        if not fp:
            return json.dumps(self.data)
        else:
            return json.dump(self.data, fp)
    
    @staticmethod
    def init_mat():
        return {}

    @staticmethod
    def append(data, prev):
        prev.data[data[0]] = data[1]
        return prev

    @staticmethod
    def materialize(data, fp = None):
        if not fp:
            return json.dumps(data)
        else:
            return json.dump(data, fp)


class ConstantStream(DataStream):
    def __init__(self, data, name, limit = -1):
        super().__init__(name)
        self.data = data
        self.iters = {}
        self.limit = limit
    
    def _next(self, op_name):
        i = self.iters[op_name][0]
        if self.limit > 0 and i >= self.limit:
            self.iters[op_name] = None
            return None
        return self.data

    def next(self, op_name, next_step = 1):
        super().next(op_name)
        return self._next(op_name)
        
    
    def skip_next(self, op_name, skip, next_step = 1):
        super().skip_next(op_name, skip, next_step)
        return self._next(op_name)
    
    @staticmethod
    def init_mat():
        return None
    
    @staticmethod
    def append(data, prev):
        return data
    
    @staticmethod
    def materialize(data):
        return data


class VideoStream(DataStream):
    def __init__(self, name, src, limit=-1, start_time = 0):
        super().__init__(name)
        self.src = src
        self.limit = limit
        self.start_time = start_time
    
    @staticmethod
    def init_mat(self, file_name, encoding, frame_rate):
        super().init_mat()
    
    @staticmethod
    def append(self, data, prev):
        super().append(data, prev)
    
    @staticmethod
    def materialize(self, data):
        return True


class CVVideoStream(VideoStream):
    def __init__(self, src, name, limit = -1):
        super().__init__(name, src, limit)
        import cv2
        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            raise CorruptedOrMissingVideo(str(self.src) + " is corrupted or missing.")
        
        self.cache = []
        self.min_frame = 0
        self.width = int(self.cap.get(3))   # float
        self.height = int(self.cap.get(4)) # float
        self.limit = limit
        self.iters = {}

    def _next(self, op_name):
        cap = self.cap
        i = self.iters[op_name][0]

        if (self.limit > 0 and i > self.limit - 1):
            self.iters[op_name] = None
            return None
        
        index = i - self.min_frame
        dif = i - (self.min_frame + len(self.cache)) + 1
        
        if dif > 0:
            for _ in range(dif):
                if cap.isOpened():
                    ret, frame = cap.read()
                    
                    if ret:
                        self.cache.append(frame)
                    else:
                        self.iters[op_name] = None
                        return None
                else:
                    self.iters[op_name] = None
                    return None
        
        frame = self.cache[index]

        min_next = float('inf')
        for key in self.iters:
            if self.iters[key][1] < min_next:
                min_next = self.iters[key][1]

        if min_next > self.min_frame:
            k = min_next - self.min_frame
            self.cache = self.cache[k:]
        return frame


    def next(self, op_name, next_step = 1):
        super().next(op_name, next_step)
        return self._next(op_name)

    def skip_next(self, op_name, skip, next_step = 1):
        super().skip_next(op_name, skip, next_step)
        return self._next(op_name)
    
    def get_cap_info(self, propId):
        """ If we currently have a VideoCapture op
        """
        if self.cap:
            return self.cap.get(propId)
        else:
            return None

    @staticmethod
    def init_mat(file_name, encoding, width, height, frame_rate):
        fourcc = cv2.VideoWriter_fourcc(*encoding)
        writer = cv2.VideoWriter(file_name,
                        fourcc,
                        frame_rate,
                        (width, height),
                        True)
        return writer
        
    @staticmethod
    def append(data, prev):
        prev.write(data)
        return prev


