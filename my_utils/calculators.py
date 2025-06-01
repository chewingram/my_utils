class claculator():
    pass


class MTP_calculator(calculator):
    def __init__(self, MTP_model)
        self.model = MTP_model
        if self.model.is_trained == False:
            raise ValueError('The MTP_model you provided is not trained!')