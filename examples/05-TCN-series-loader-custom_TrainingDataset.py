import numpy as np
import torch
from darts import TimeSeries

# for reproducibility
torch.manual_seed(1)
np.random.seed(1)

from darts.utils.data import PastCovariatesTrainingDataset, InferenceDataset, PastCovariatesInferenceDataset

from darts.models import (
    TCNModel,
)

'''
params
'''
nlag = 10
npred = 5
nf=4
nc=2

'''
create the actual dataset - just numbered rows for debugging
'''
def make_array(shape_tuple):

    array = np.ones(shape=shape_tuple).T
    for i in range(0,shape_tuple[0]):
        array[:,i]*=i

    return array.T

# training dataset
class MyDataset(PastCovariatesTrainingDataset):

    def __init__(self, nlag, npred, nf, nc):
        """
        Abstract class for a PastCovariatesTorchModel training dataset. It contains 3-tuples of
        `(past_target, past_covariate, future_target)` `np.ndarray`.
        The covariates are optional and can be `None`.
        """
        super().__init__()

        # create a file to index mapping - multivar dim
        self.nlag = nlag
        self.npred = npred
        self.nrow = nlag+npred
        self.nf = nf
        self.nc = nc

        # fake random data for now
        self.data = []

        def _make_data(input_dim_AR, input_dim_cov, output_dim):
            # make the datapoints
            X_AR = make_array(input_dim_AR)
            X_cov = make_array(input_dim_cov)
            X_cov[:,1:] += nc
            y = make_array(output_dim)
            return (X_AR, X_cov, y)

        X_AR, X_cov, y = _make_data((nlag, nf), (nlag, nc), (nlag,nf))
        self.data.append((X_AR, X_cov, y))
        X_AR, X_cov, y = _make_data((nlag+1, nf), (nlag+1, nc), (nlag+1, nf))
        X_AR += 100; X_cov += 100; y+=100
        self.data.append((X_AR[0:-1,:], X_cov[0:-1,:], y[0:-1,:]))
        self.data.append((X_AR[1:, :], X_cov[1:, :], y[1:, :]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

data_train = MyDataset(nlag, npred, nf, nc)
data_val = MyDataset(nlag, npred, nf, nc)

class MyDatasetTest(InferenceDataset):

    def __init__(self, ds):
        """
        Abstract class for a PastCovariatesTorchModel training dataset. It contains 3-tuples of
        `(past_target, past_covariate, future_target)` `np.ndarray`.
        The covariates are optional and can be `None`.
        """
        super().__init__()

        # just copy a subset of training data
        self.data = [ds.data[0], ds.data[-1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# just copy training data - purely a mechanical test
data_test = MyDatasetTest(data_train)

model = TCNModel(
    input_chunk_length=nlag,
    output_chunk_length=npred,
    n_epochs=2,
    dropout=0.1,
    dilation_base=2,
    weight_norm=True,
    kernel_size=5,
    num_filters=9,
    random_state=0,
)

model.fit_from_dataset(data_train, data_val)
# the following was needed but not anymore for some reason...
# model._fit_called = True

test_series = []
test_series_cov = []
for i in range(2):
    test_series.append(TimeSeries.from_values(np.random.randn(nlag+1, nf)).astype(np.float32))
    test_series_cov.append(TimeSeries.from_values(np.random.randn(nlag+1, nc)).astype(np.float32))

p=model.predict(n=npred, series=test_series, past_covariates=test_series_cov)
print(p)

p_ds = model.predict_from_dataset(npred, data_test)
r"""
The above errors with
2022-05-03 19:20:14 main_logger ERROR: ValueError: expected type <class 'darts.utils.data.inference_dataset.PastCovariatesInferenceDataset'>, got: <class '__main__.MyDatasetTest'>
Traceback (most recent call last):
  File "C:/repos/darts/examples/05-TCN-series-loader-custom_TrainingDataset.py", line 142, in <module>
    p_ds = model.predict_from_dataset(npred, data_test)
  File "C:\repos\darts\darts\utils\torch.py", line 70, in decorator
    return decorated(self, *args, **kwargs)
  File "C:\repos\darts\darts\models\forecasting\torch_forecasting_model.py", line 1176, in predict_from_dataset
    self._verify_inference_dataset_type(input_series_dataset)
  File "C:\repos\darts\darts\models\forecasting\torch_forecasting_model.py", line 1619, in _verify_inference_dataset_type
    _raise_if_wrong_type(inference_dataset, PastCovariatesInferenceDataset)
  File "C:\repos\darts\darts\models\forecasting\torch_forecasting_model.py", line 1473, in _raise_if_wrong_type
    raise_if_not(isinstance(obj, exp_type), msg.format(exp_type, type(obj)))
  File "C:\repos\darts\darts\logging.py", line 84, in raise_if_not
    raise ValueError(message)
ValueError: expected type <class 'darts.utils.data.inference_dataset.PastCovariatesInferenceDataset'>, got: <class '__main__.MyDatasetTest'>
"""

print(p_ds)

print("done")
