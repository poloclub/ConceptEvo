class LayerActByExPatch:
    """
    Layer activation of a base model with example patches
    """

    def __init__(self, args, data_path):
        self.args = args
        self.data_path = data_path

        if 'pretrained' in self.args.model_nickname:
            self.model_nickname = self.args.model_nickname
        else:
            self.model_nickname = f'{self.args.model_nickname}_{self.args.epoch}'