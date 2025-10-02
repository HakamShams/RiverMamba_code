#!/bin/sh

# you can download the data directly from [https://doi.org/10.60507/FK2/T8QYWE]

mkdir -p GloFAS_Reanalysis_Global

# statistics for training
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13831 -O GloFAS_Reanalysis_Global/GloFAS_statistics_train.json

# download each year separately
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13785 -O GloFAS_Reanalysis_Global/1979.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13786 -O GloFAS_Reanalysis_Global/1980.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13787 -O GloFAS_Reanalysis_Global/1981.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13788 -O GloFAS_Reanalysis_Global/1982.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13789 -O GloFAS_Reanalysis_Global/1983.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13790 -O GloFAS_Reanalysis_Global/1984.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13791 -O GloFAS_Reanalysis_Global/1985.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13792 -O GloFAS_Reanalysis_Global/1986.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13793 -O GloFAS_Reanalysis_Global/1987.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13794 -O GloFAS_Reanalysis_Global/1988.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13795 -O GloFAS_Reanalysis_Global/1989.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13796 -O GloFAS_Reanalysis_Global/1990.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13797 -O GloFAS_Reanalysis_Global/1991.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13798 -O GloFAS_Reanalysis_Global/1992.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13799 -O GloFAS_Reanalysis_Global/1993.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13800 -O GloFAS_Reanalysis_Global/1994.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13801 -O GloFAS_Reanalysis_Global/1995.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13802 -O GloFAS_Reanalysis_Global/1996.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13803 -O GloFAS_Reanalysis_Global/1997.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13804 -O GloFAS_Reanalysis_Global/1998.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13805 -O GloFAS_Reanalysis_Global/1999.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13806 -O GloFAS_Reanalysis_Global/2000.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13807 -O GloFAS_Reanalysis_Global/2001.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13808 -O GloFAS_Reanalysis_Global/2002.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13809 -O GloFAS_Reanalysis_Global/2003.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13810 -O GloFAS_Reanalysis_Global/2004.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13811 -O GloFAS_Reanalysis_Global/2005.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13812 -O GloFAS_Reanalysis_Global/2006.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13813 -O GloFAS_Reanalysis_Global/2007.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13814 -O GloFAS_Reanalysis_Global/2008.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13815 -O GloFAS_Reanalysis_Global/2009.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13816 -O GloFAS_Reanalysis_Global/2010.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13817 -O GloFAS_Reanalysis_Global/2011.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13818 -O GloFAS_Reanalysis_Global/2012.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13819 -O GloFAS_Reanalysis_Global/2013.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13820 -O GloFAS_Reanalysis_Global/2014.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13821 -O GloFAS_Reanalysis_Global/2015.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13822 -O GloFAS_Reanalysis_Global/2016.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13823 -O GloFAS_Reanalysis_Global/2017.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13824 -O GloFAS_Reanalysis_Global/2018.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13825 -O GloFAS_Reanalysis_Global/2019.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13826 -O GloFAS_Reanalysis_Global/2020.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13827 -O GloFAS_Reanalysis_Global/2021.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13828 -O GloFAS_Reanalysis_Global/2022.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13829 -O GloFAS_Reanalysis_Global/2023.7z
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13830 -O GloFAS_Reanalysis_Global/2024.7z

