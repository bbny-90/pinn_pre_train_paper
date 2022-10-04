
class AuxilaryTaskScheduler():
    def __init__(self, params: dict) -> None:
        self.method = params["method"]
        if self.method == "stepwise":
            self.offset_epoch = params["offset_epoch"]
            self.reduction_factor = params["reduction_factor"]
            self.step_size = params["step_size"]
        elif self.method == "jump":
            self.offset_epoch = params["offset_epoch"]
        else:
            raise NotImplementedError(self.method)
    
    def __call__(self, curent_penalty, epoch) -> float:
        if self.method == "stepwise":
            if epoch > self.offset_epoch and\
               epoch % self.step_size == 0:
                curent_penalty /= self.reduction_factor
        elif self.method == "jump":
            if epoch > self.offset_epoch:
                curent_penalty = 0.
        else:
            raise NotADirectoryError(self.method)
        return curent_penalty