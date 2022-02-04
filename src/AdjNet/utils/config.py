from __future__ import print_function
import os
import platform

class Config_setup(object):
    """docstring for Config"""
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Config_setup, cls).__call__(*args, **kwargs)
        else:
            cls._instances[cls].__init__(*args, **kwargs)
        return cls._instances[cls]

    def __init__(self, DATAPATH=None):
        super(Config, self).__init__()

        if DATAPATH is None:
            DATAPATH = os.environ.get('DATAPATH')
            if DATAPATH is None:
                if platform.system() == "Windows" or platform.system() == "Linux":
                    # DATAPATH = "D:/data/traffic_flow"
                # elif platform.system() == "Linux":
                    DATAPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..', 'data')
                else:
                    print("Unsupported/Unknown OS: ", platform.system, "please set DATAPATH")
        self.DATAPATH = DATAPATH
        
class Config(metaclass=Config_setup):
    pass