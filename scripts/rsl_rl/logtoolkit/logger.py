from .logger_lr import LoggerLR
from .logger_limit import LoggerLimits

class Logger:
    def __init__(self, env):
        self.log = LoggerLR(env)
        #self.log = LoggerLimits(env)

    def log_step(self, actions, extra):
        self.log.log_step(actions, extra)
