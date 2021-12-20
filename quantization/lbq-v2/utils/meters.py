class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, top_k=(1,)):
    """Computes the precision@k for the specified values of k"""
    max_k = max(top_k)
    batch_size = target.size(0)
    _, pred = output.topk(max_k, 1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Meter(object):
    """Meter is to keep track of statistics along steps.
    Meters cache values for purpose like printing average values.
    Meters can be flushed to log files (i.e. TensorBoard) regularly.

    Args:
        name (str): the name of meter

    """
    def __init__(self, name):
        self.name = name
        self.steps = 0
        self.reset()

    def __repr__(self):
        return '{}: {!r}'.format(self.__class__, self.__dict__)

    def reset(self):
        self.values = []

    def cache(self, value, pstep=1):
        self.steps += pstep
        self.values.append(value)

    def cache_list(self, value_list, pstep=1):
        self.steps += pstep
        self.values += value_list

    def flush(self, value, reset=True):
        pass


class ScalarMeter(Meter):
    """ScalarMeter records scalar over steps.

    """
    def __init__(self, name):
        super(ScalarMeter, self).__init__(name)

    def flush(self, value, step=-1, reset=True):
        if reset:
            self.reset()


def flush_scalar_meters(meters, method='avg'):
    """Docstring for flush_scalar_meters"""
    results = {}
    assert isinstance(meters, dict), "meters should be a dict."
    for name, meter in meters.items():
        if not isinstance(meter, ScalarMeter):
            #continue
            results[name] = meter
            continue
        if method == 'avg':
            value = sum(meter.values) / len(meter.values)
        elif method == 'sum':
            value = sum(meter.values)
        elif method == 'max':
            value = max(meter.values)
        elif method == 'min':
            value = min(meter.values)
        else:
            raise NotImplementedError(
                'flush method: {} is not yet implemented.'.format(method))
        results[name] = float(format(value, '.3f'))
        meter.flush(value)
    return results
