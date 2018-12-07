class ModelHandler:

    def __init__(self,
                 model_name,
                 model_factory=None):
        self.model_name=model_name         
        self._model_factory=ModelHandler.generate_raise(NotImplementedError('Model factory not set')) if not model_factory else model_factory
        self._model=ModelHandler.generate_raise(ValueError('Model not created'))

    @staticmethod
    def generate_raise(e):
        def raiser():
            raise e
        return raiser
    
    @property
    def model_factory(self):
        return self._model_factory
    
    @model_factory.setter
    def model_factory(self, model_factory):
        self._model_factory=model_factory
        
    @property
    def model_parameters(self):
        import inspect
        return tuple(inspect.getcallargs(self.model_factory).keys())
        
    @property
    def model(self):
        return self._model
        
    def autofill_params(self, params_dict):
        '''
        Auto fill generator with parameters from parmas_dict
        returns curried generator
        '''
        model_params = {k:v for k,v in params_dict.items() if k in self.model_parameters}
        self._model=self.model_factory(**model_params)
        return self._model