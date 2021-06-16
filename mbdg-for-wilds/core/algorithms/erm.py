

class ERM:
    def __init__(self, model, criterion, args):
        self._model = model
        self._criterion = criterion
        self._fname = f'erm-lr-{args.lr}-trial-{args.trial_index}'

    def __call__(self, imgs, labels):
        return self.step(imgs, labels)

    def __str__(self):
        return 'ERM'

    @property
    def fname(self):
        return self._fname

    def step(self, imgs, labels):
        
        output = self._model(imgs)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(labels.view_as(pred)).sum().item()
        loss = self._criterion(output, labels)

        return loss, correct
