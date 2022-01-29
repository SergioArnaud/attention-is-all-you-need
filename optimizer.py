class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self.optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.step_n = 0


    def step(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self.optimizer.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
        
    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self.optimizer.zero_grad()


    def _get_lr_scale(self):
        d_model = self.d_model
        step_n, n_warmup_steps = self.step_n, self.n_warmup_steps
        return (d_model ** -0.5) * min(step_n ** (-0.5), step_n * n_warmup_steps ** (-1.5))


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.step_n += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr