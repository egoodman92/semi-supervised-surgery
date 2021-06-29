from collections import defaultdict 

class BatchLogger(): 
    def __init__(self): 
        self.filename = defaultdict(int) 
        self.videoID = defaultdict(int) 
        self.index = None 
#        self.tool = {'bovie': [0, 0], 'scalpel': [0, 0], 'forceps': [0, 0], 'needledriver': [0, 0], 'background': [0, 0]} #defaultdict(int) 
#        self.tool_frame = {'bovie': [0, 0], 'scalpel': [0, 0], 'forceps': [0, 0], 'needledriver': [0, 0], 'background': [0, 0]} #defaultdict(int) 
        self.tool = {'bovie': [0, 0], 'scalpel': [0, 0], 'forceps': [0, 0], 'needledriver': [0, 0], 'background': [0, 0], 'hand': [0, 0]}
        self.tool_frame = {'bovie': [0, 0], 'scalpel': [0, 0], 'forceps': [0, 0], 'needledriver': [0, 0], 'background': [0, 0], 'hand': [0, 0]}


        self.background = {"tools_present":0, "tools_absent":0} 
        
    def dictify(self): 
        self.filename = dict(self.filename) 
        self.videoID = dict(self.videoID) 
        if self.index is not None: 
            self.index = dict(self.index) 
        self.tool = dict(self.tool) 
        self.tool_frame = dict(self.tool_frame) 
        self.background = dict(self.background) 
        
        
    def undictify(self): 
        self.filename = defaultdict(int, self.filename) 
        self.videoID = defaultdict(int, self.videoID) 
