#!/bin/sh

# you can download the data directly from [https://doi.org/10.60507/FK2/T8QYWE]

mkdir -p ECMWF_HRES_Global

# statistics for training
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13599 -O ECMWF_HRES_Global/hres_statistics_train.json

# download each year separately
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13554 -O ECMWF_HRES_Global/2010.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13555 -O ECMWF_HRES_Global/2010.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13556 -O ECMWF_HRES_Global/2010.7z.003

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13557 -O ECMWF_HRES_Global/2011.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13558 -O ECMWF_HRES_Global/2011.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13559 -O ECMWF_HRES_Global/2011.7z.003

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13560 -O ECMWF_HRES_Global/2012.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13561 -O ECMWF_HRES_Global/2012.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13562 -O ECMWF_HRES_Global/2012.7z.003

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13563 -O ECMWF_HRES_Global/2013.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13564 -O ECMWF_HRES_Global/2013.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13565 -O ECMWF_HRES_Global/2013.7z.003

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13566 -O ECMWF_HRES_Global/2014.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13567 -O ECMWF_HRES_Global/2014.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13568 -O ECMWF_HRES_Global/2014.7z.003

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13569 -O ECMWF_HRES_Global/2015.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13570 -O ECMWF_HRES_Global/2015.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13571 -O ECMWF_HRES_Global/2015.7z.003

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13572 -O ECMWF_HRES_Global/2016.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13573 -O ECMWF_HRES_Global/2016.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13574 -O ECMWF_HRES_Global/2016.7z.003

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13575 -O ECMWF_HRES_Global/2017.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13576 -O ECMWF_HRES_Global/2017.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13577 -O ECMWF_HRES_Global/2017.7z.003

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13578 -O ECMWF_HRES_Global/2018.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13579 -O ECMWF_HRES_Global/2018.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13580 -O ECMWF_HRES_Global/2018.7z.003

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13581 -O ECMWF_HRES_Global/2019.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13582 -O ECMWF_HRES_Global/2019.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13583 -O ECMWF_HRES_Global/2019.7z.003

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13584 -O ECMWF_HRES_Global/2020.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13585 -O ECMWF_HRES_Global/2020.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13586 -O ECMWF_HRES_Global/2020.7z.003

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13587 -O ECMWF_HRES_Global/2021.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13588 -O ECMWF_HRES_Global/2021.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13589 -O ECMWF_HRES_Global/2021.7z.003

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13590 -O ECMWF_HRES_Global/2022.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13591 -O ECMWF_HRES_Global/2022.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13592 -O ECMWF_HRES_Global/2022.7z.003

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13593 -O ECMWF_HRES_Global/2023.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13594 -O ECMWF_HRES_Global/2023.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13595 -O ECMWF_HRES_Global/2023.7z.003

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13596 -O ECMWF_HRES_Global/2024.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13597 -O ECMWF_HRES_Global/2024.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13598 -O ECMWF_HRES_Global/2024.7z.003


