class EarlyStopper:
    """Stops training if validation loss doesn't improve for `patience` epochs."""

    def __init__(self, patience: int, min_delta: float):
        self._patience = patience
        self._min_delta = min_delta
        self._counter = 0
        self._min_validation_loss = float('inf')

    def check_early_stopping(self, validation_loss: float) -> bool:
        if validation_loss < self._min_validation_loss - self._min_delta:
            self._min_validation_loss = validation_loss
            self._counter = 0
        else:
            self._counter += 1

        return self._counter >= self._patience
