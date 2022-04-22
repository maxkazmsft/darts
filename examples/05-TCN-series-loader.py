'''
Loads data just like darts using series - lists of series can be memory mapped to disk if one doesn't have enough RAM
https://stackoverflow.com/questions/241141/python-lazy-list
'''

import numpy as np
import torch
from darts import TimeSeries
from pytorch_lightning import Trainer

# for reproducibility
from darts.utils.data.shifted_dataset import GenericShiftedDataset
from darts.utils.data.utils import CovariateType

torch.manual_seed(1)
np.random.seed(1)

from darts.models import (
    RNNModel,
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


train_series = []
train_series_cov = []
nrow = nlag+npred

train_series.append(TimeSeries.from_values(make_array((nrow, nf))).astype(np.float32))
train_series_cov.append(TimeSeries.from_values(make_array((nrow, nc))).astype(np.float32))

# offset second series by 100 to tell the difference in debugger
train_series.append(TimeSeries.from_values(make_array((nrow+1, nf))+100).astype(np.float32))
train_series_cov.append(TimeSeries.from_values(make_array((nrow+1, nc))+100).astype(np.float32))

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

ds = GenericShiftedDataset(
    target_series=train_series,
    covariates=train_series_cov,
    input_chunk_length=nlag,
    output_chunk_length=nlag,
    shift=npred,
    shift_covariates=False,
    max_samples_per_ts=None,
    covariate_type=CovariateType.PAST,
)
'''
DEBUG from above shows that there are 4 and not 2 datapoints, and they're shifted by 1 instead of nlag (look at the 
data coming from the second series). Output is below. 

ds.__len__()
Out[3]: 4
ds[0]
Out[4]: 
(array([[0., 0., 0., 0.],
        [1., 1., 1., 1.],
        [2., 2., 2., 2.],
        [3., 3., 3., 3.],
        [4., 4., 4., 4.],
        [5., 5., 5., 5.],
        [6., 6., 6., 6.],
        [7., 7., 7., 7.],
        [8., 8., 8., 8.],
        [9., 9., 9., 9.]], dtype=float32),
 array([[0., 0.],
        [1., 1.],
        [2., 2.],
        [3., 3.],
        [4., 4.],
        [5., 5.],
        [6., 6.],
        [7., 7.],
        [8., 8.],
        [9., 9.]], dtype=float32),
 array([[ 5.,  5.,  5.,  5.],
        [ 6.,  6.,  6.,  6.],
        [ 7.,  7.,  7.,  7.],
        [ 8.,  8.,  8.,  8.],
        [ 9.,  9.,  9.,  9.],
        [10., 10., 10., 10.],
        [11., 11., 11., 11.],
        [12., 12., 12., 12.],
        [13., 13., 13., 13.],
        [14., 14., 14., 14.]], dtype=float32))
ds[1]
Out[5]: 
(array([[0., 0., 0., 0.],
        [1., 1., 1., 1.],
        [2., 2., 2., 2.],
        [3., 3., 3., 3.],
        [4., 4., 4., 4.],
        [5., 5., 5., 5.],
        [6., 6., 6., 6.],
        [7., 7., 7., 7.],
        [8., 8., 8., 8.],
        [9., 9., 9., 9.]], dtype=float32),
 array([[0., 0.],
        [1., 1.],
        [2., 2.],
        [3., 3.],
        [4., 4.],
        [5., 5.],
        [6., 6.],
        [7., 7.],
        [8., 8.],
        [9., 9.]], dtype=float32),
 array([[ 5.,  5.,  5.,  5.],
        [ 6.,  6.,  6.,  6.],
        [ 7.,  7.,  7.,  7.],
        [ 8.,  8.,  8.,  8.],
        [ 9.,  9.,  9.,  9.],
        [10., 10., 10., 10.],
        [11., 11., 11., 11.],
        [12., 12., 12., 12.],
        [13., 13., 13., 13.],
        [14., 14., 14., 14.]], dtype=float32))
ds[2]
Out[6]: 
(array([[101., 101., 101., 101.],
        [102., 102., 102., 102.],
        [103., 103., 103., 103.],
        [104., 104., 104., 104.],
        [105., 105., 105., 105.],
        [106., 106., 106., 106.],
        [107., 107., 107., 107.],
        [108., 108., 108., 108.],
        [109., 109., 109., 109.],
        [110., 110., 110., 110.]], dtype=float32),
 array([[101., 101.],
        [102., 102.],
        [103., 103.],
        [104., 104.],
        [105., 105.],
        [106., 106.],
        [107., 107.],
        [108., 108.],
        [109., 109.],
        [110., 110.]], dtype=float32),
 array([[106., 106., 106., 106.],
        [107., 107., 107., 107.],
        [108., 108., 108., 108.],
        [109., 109., 109., 109.],
        [110., 110., 110., 110.],
        [111., 111., 111., 111.],
        [112., 112., 112., 112.],
        [113., 113., 113., 113.],
        [114., 114., 114., 114.],
        [115., 115., 115., 115.]], dtype=float32))
ds[3]
Out[7]: 
(array([[100., 100., 100., 100.],
        [101., 101., 101., 101.],
        [102., 102., 102., 102.],
        [103., 103., 103., 103.],
        [104., 104., 104., 104.],
        [105., 105., 105., 105.],
        [106., 106., 106., 106.],
        [107., 107., 107., 107.],
        [108., 108., 108., 108.],
        [109., 109., 109., 109.]], dtype=float32),
 array([[100., 100.],
        [101., 101.],
        [102., 102.],
        [103., 103.],
        [104., 104.],
        [105., 105.],
        [106., 106.],
        [107., 107.],
        [108., 108.],
        [109., 109.]], dtype=float32),
 array([[105., 105., 105., 105.],
        [106., 106., 106., 106.],
        [107., 107., 107., 107.],
        [108., 108., 108., 108.],
        [109., 109., 109., 109.],
        [110., 110., 110., 110.],
        [111., 111., 111., 111.],
        [112., 112., 112., 112.],
        [113., 113., 113., 113.],
        [114., 114., 114., 114.]], dtype=float32))
'''

'''
PROBLEMS:
The length of the first training series is nlag+npred, so we expect a single data point to be created from that training series, where dependent variable has 
dimensionality npred x nf, but from debugged output below you can see that we get a duplicate of the same training point twice, 
and dependent variable has dimensionality nlag x nf instead of npred x nf.
 
The length of the second training series is nlag+npred+1, so we can move the sliding window by 1 datapoint and create 2 more training points
for a grand total of 1+2=3 datapoints IF input_chunk_length=1, but actually input_chunk_length=nlag=10, so we would actually also produce only a single
datapoint from the second series (no space to move sliding window by nlag=10). However, from debugger output, you can see that __getitem__(2)
and __getitem__(3) move by input_chunk_length=1 and have the same problem with dependent variable dimensionality.

Inside the debugger, darts/models/forecasting/torch_forecasting_model.py on line 771 has:
train_dataset.__len__()
Out[3]: 4

More output from debugger
train_dataset.__getitem__(0)
Out[4]: 
(array([[0., 0., 0., 0.],
        [1., 1., 1., 1.],
        [2., 2., 2., 2.],
        [3., 3., 3., 3.],
        [4., 4., 4., 4.],
        [5., 5., 5., 5.],
        [6., 6., 6., 6.],
        [7., 7., 7., 7.],
        [8., 8., 8., 8.],
        [9., 9., 9., 9.]], dtype=float32),
 array([[0., 0.],
        [1., 1.],
        [2., 2.],
        [3., 3.],
        [4., 4.],
        [5., 5.],
        [6., 6.],
        [7., 7.],
        [8., 8.],
        [9., 9.]], dtype=float32),
 array([[ 5.,  5.,  5.,  5.],
        [ 6.,  6.,  6.,  6.],
        [ 7.,  7.,  7.,  7.],
        [ 8.,  8.,  8.,  8.],
        [ 9.,  9.,  9.,  9.],
        [10., 10., 10., 10.],
        [11., 11., 11., 11.],
        [12., 12., 12., 12.],
        [13., 13., 13., 13.],
        [14., 14., 14., 14.]], dtype=float32))
train_dataset.__getitem__(1)
Out[5]: 
(array([[0., 0., 0., 0.],
        [1., 1., 1., 1.],
        [2., 2., 2., 2.],
        [3., 3., 3., 3.],
        [4., 4., 4., 4.],
        [5., 5., 5., 5.],
        [6., 6., 6., 6.],
        [7., 7., 7., 7.],
        [8., 8., 8., 8.],
        [9., 9., 9., 9.]], dtype=float32),
 array([[0., 0.],
        [1., 1.],
        [2., 2.],
        [3., 3.],
        [4., 4.],
        [5., 5.],
        [6., 6.],
        [7., 7.],
        [8., 8.],
        [9., 9.]], dtype=float32),
 array([[ 5.,  5.,  5.,  5.],
        [ 6.,  6.,  6.,  6.],
        [ 7.,  7.,  7.,  7.],
        [ 8.,  8.,  8.,  8.],
        [ 9.,  9.,  9.,  9.],
        [10., 10., 10., 10.],
        [11., 11., 11., 11.],
        [12., 12., 12., 12.],
        [13., 13., 13., 13.],
        [14., 14., 14., 14.]], dtype=float32))
train_dataset.__getitem__(2)
Out[6]: 
(array([[101., 101., 101., 101.],
        [102., 102., 102., 102.],
        [103., 103., 103., 103.],
        [104., 104., 104., 104.],
        [105., 105., 105., 105.],
        [106., 106., 106., 106.],
        [107., 107., 107., 107.],
        [108., 108., 108., 108.],
        [109., 109., 109., 109.],
        [110., 110., 110., 110.]], dtype=float32),
 array([[101., 101.],
        [102., 102.],
        [103., 103.],
        [104., 104.],
        [105., 105.],
        [106., 106.],
        [107., 107.],
        [108., 108.],
        [109., 109.],
        [110., 110.]], dtype=float32),
 array([[106., 106., 106., 106.],
        [107., 107., 107., 107.],
        [108., 108., 108., 108.],
        [109., 109., 109., 109.],
        [110., 110., 110., 110.],
        [111., 111., 111., 111.],
        [112., 112., 112., 112.],
        [113., 113., 113., 113.],
        [114., 114., 114., 114.],
        [115., 115., 115., 115.]], dtype=float32))
train_dataset.__getitem__(3)
Out[7]: 
(array([[100., 100., 100., 100.],
        [101., 101., 101., 101.],
        [102., 102., 102., 102.],
        [103., 103., 103., 103.],
        [104., 104., 104., 104.],
        [105., 105., 105., 105.],
        [106., 106., 106., 106.],
        [107., 107., 107., 107.],
        [108., 108., 108., 108.],
        [109., 109., 109., 109.]], dtype=float32),
 array([[100., 100.],
        [101., 101.],
        [102., 102.],
        [103., 103.],
        [104., 104.],
        [105., 105.],
        [106., 106.],
        [107., 107.],
        [108., 108.],
        [109., 109.]], dtype=float32),
 array([[105., 105., 105., 105.],
        [106., 106., 106., 106.],
        [107., 107., 107., 107.],
        [108., 108., 108., 108.],
        [109., 109., 109., 109.],
        [110., 110., 110., 110.],
        [111., 111., 111., 111.],
        [112., 112., 112., 112.],
        [113., 113., 113., 113.],
        [114., 114., 114., 114.]], dtype=float32))

'''

model.fit(series=train_series, past_covariates=train_series_cov,
          trainer=Trainer(max_epochs=2))

'''
Now test series behaves correctly - it produces 2 predictions, npred x nf, because with a lag of 10 we can only produce a single
data point from each series of length nlag+1 (not stepping by 1). 
'''
test_series = []
test_series_cov = []
for i in range(2):
    test_series.append(TimeSeries.from_values(np.random.randn(nlag+1, nf)).astype(np.float32))
    test_series_cov.append(TimeSeries.from_values(np.random.randn(nlag+1, nc)).astype(np.float32))
p=model.predict(n=npred, series=test_series, past_covariates=test_series_cov)

print(p)

print("done")
