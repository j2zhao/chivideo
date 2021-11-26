"""This file is part of DeepLens which is released under MIT License and 
is copyrighted by the University of Chicago. This project is developed by
the database group (chidata).

pipeline.py defines the main data structures used in deeplens's pipeline. It defines a
video input stream as well as operators that can transform this stream.
"""
from chivideo.streams import *
import copy

#sources video from the default camera
DEFAULT_CAMERA = 0

class Operator():
    """An operator defines consumes an iterator over frames
    and produces and iterator over frames. The Operator class
    is the abstract class of all pipeline components in dlcv.
    """
    # need to initialize appropriate dstreams
    def __init__(self, name, input_names, output_names):
        self.name = name
        self.input_names = input_names
        self.output_names = output_names
        self.istreams = {}
        self.ostreams = {}
        self.cache = Cache()

    def next(self):
        raise NotImplemented()

    # def skip_next(self):
    #     raise NotImplemented()
        
    #binds previous operators and dstreams to the current stream
    def init_inputs(self, streams):
        if streams.keys() != self.input_names:
            raise ValueError()
        self.istreams = streams
        for stream in self.istreams:
            self.istreams[stream].add_iter(self.name)

    def change_input(self, stream):
        if stream.name not in self.input_names:
            raise ValueError()
        self.istreams[stream.name] = stream
        stream.add_iter(self.name)
    
    def init_outputs(self):
        for stream in self.ostreams:
            self.ostreams[stream].set_op(self)        
        return self.ostreams

    def clear_cache(self):
        self.cache.clear_all()

class Cache():
    ''' Saves state data in operators
    '''
    def __init__(self):
        self.data = {}
    
    def get(self, key):
        if key in self.data:
            return self.data[key]
        else:
            return None

    def set(self, key, val):
        self.data[key] = val

    def clear(self, key):
        del self.data[key]
    
    def clear_all(self):
        self.data = {}

class Pipeline():
    """ Creates a DAG graph for pipeline purposes
    """
    def __init__(self, istreams = {}):
        self.dstreams = {}
        self.operators = {}
        self.istreams = istreams
        self.fops = {}
        for i in istreams:
            self.fops[i] = []

    def get_operators(self):
        return self.operators

    def get_streams(self, snames):
        output = {}
        for n in snames:
            if n in self.dstreams:
                output[n] = self.dstreams[n]
            elif n in self.istreams:
                output[n] = self.istreams[n]
            else:
                raise ValueError()
        return output
    
    def get_outputs(self, snames, keep_all = True):
        output = []
        for n in snames:
            if n in self.dstreams:
                output.append(self.dstreams[n])
                self.dstreams[n].set_keep_all()
            elif n in self.istreams:
                output.append(self.istreams[n])
                self.dstreams[n].set_keep_all()
            else:
                raise ValueError()
        return tuple(output)

    def get_istreams(self):
        return self.istreams

    def add_istreams(self, istreams):
        self.istreams.update(istreams)

    def add_operator(self, op):
        keys = op.input_names
        inputs = {}
        
        _inputs = []
        for k in keys:
            if k in self.istreams.keys():
                inputs.append(self.istreams[k])
                _inputs.append(k)
            elif k in self.dstreams.keys():
                inputs.append(self.dstreams[k])
            else:
                raise ValueError() # change error types later
        
        for k in op.output_names:
            if k in self.istreams.keys():
                raise ValueError()
            if k in self.dstreams.keys():
                raise ValueError()

        
        nstreams = op.init_outputs()
        op.init_inputs(inputs)
        self.dstreams.update(nstreams)
        self.operators[op.name] = op
        for k in _inputs:
            self.fops[k].append(op)
    
    def reset(self, istreams = {}, check_position = {}, clear_cache = True):

        for key in istreams:
            for op in self.fops[key]:
                op.change_input(istreams[key])

        self.istreams.update(istreams)

        for key, position in check_position:
            self.istreams[key].update_iters(position)
        
        for key in self.dstreams:
            self.dstreams[key].clear()

        if clear_cache:
            for op in self.operators:
                self.operators[op].clear_cache()


