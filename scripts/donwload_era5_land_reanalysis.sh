#!/bin/sh

# you can download the data directly from [https://doi.org/10.60507/FK2/T8QYWE]

mkdir -p ERA5-Land_Reanalysis_Global

# statistics for training
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13784 -O ERA5-Land_Reanalysis_Global/ERA5_Land_statistics_train.json

# download each year separately
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13600 -O ERA5-Land_Reanalysis_Global/1979.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13601 -O ERA5-Land_Reanalysis_Global/1979.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13602 -O ERA5-Land_Reanalysis_Global/1979.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13603 -O ERA5-Land_Reanalysis_Global/1979.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13604 -O ERA5-Land_Reanalysis_Global/1980.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13605 -O ERA5-Land_Reanalysis_Global/1980.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13606 -O ERA5-Land_Reanalysis_Global/1980.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13607 -O ERA5-Land_Reanalysis_Global/1980.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13608 -O ERA5-Land_Reanalysis_Global/1981.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13609 -O ERA5-Land_Reanalysis_Global/1981.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13610 -O ERA5-Land_Reanalysis_Global/1981.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13611 -O ERA5-Land_Reanalysis_Global/1981.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13612 -O ERA5-Land_Reanalysis_Global/1982.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13613 -O ERA5-Land_Reanalysis_Global/1982.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13614 -O ERA5-Land_Reanalysis_Global/1982.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13615 -O ERA5-Land_Reanalysis_Global/1982.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13616 -O ERA5-Land_Reanalysis_Global/1983.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13617 -O ERA5-Land_Reanalysis_Global/1983.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13618 -O ERA5-Land_Reanalysis_Global/1983.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13619 -O ERA5-Land_Reanalysis_Global/1983.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13620 -O ERA5-Land_Reanalysis_Global/1984.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13621 -O ERA5-Land_Reanalysis_Global/1984.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13622 -O ERA5-Land_Reanalysis_Global/1984.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13623 -O ERA5-Land_Reanalysis_Global/1984.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13624 -O ERA5-Land_Reanalysis_Global/1985.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13625 -O ERA5-Land_Reanalysis_Global/1985.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13626 -O ERA5-Land_Reanalysis_Global/1985.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13627 -O ERA5-Land_Reanalysis_Global/1985.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13628 -O ERA5-Land_Reanalysis_Global/1986.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13629 -O ERA5-Land_Reanalysis_Global/1986.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13630 -O ERA5-Land_Reanalysis_Global/1986.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13631 -O ERA5-Land_Reanalysis_Global/1986.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13632 -O ERA5-Land_Reanalysis_Global/1987.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13633 -O ERA5-Land_Reanalysis_Global/1987.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13634 -O ERA5-Land_Reanalysis_Global/1987.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13635 -O ERA5-Land_Reanalysis_Global/1987.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13636 -O ERA5-Land_Reanalysis_Global/1988.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13637 -O ERA5-Land_Reanalysis_Global/1988.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13638 -O ERA5-Land_Reanalysis_Global/1988.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13639 -O ERA5-Land_Reanalysis_Global/1988.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13640 -O ERA5-Land_Reanalysis_Global/1989.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13641 -O ERA5-Land_Reanalysis_Global/1989.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13642 -O ERA5-Land_Reanalysis_Global/1989.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13643 -O ERA5-Land_Reanalysis_Global/1989.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13644 -O ERA5-Land_Reanalysis_Global/1990.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13645 -O ERA5-Land_Reanalysis_Global/1990.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13646 -O ERA5-Land_Reanalysis_Global/1990.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13647 -O ERA5-Land_Reanalysis_Global/1990.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13648 -O ERA5-Land_Reanalysis_Global/1991.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13649 -O ERA5-Land_Reanalysis_Global/1991.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13650 -O ERA5-Land_Reanalysis_Global/1991.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13651 -O ERA5-Land_Reanalysis_Global/1991.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13652 -O ERA5-Land_Reanalysis_Global/1992.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13653 -O ERA5-Land_Reanalysis_Global/1992.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13654 -O ERA5-Land_Reanalysis_Global/1992.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13655 -O ERA5-Land_Reanalysis_Global/1992.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13656 -O ERA5-Land_Reanalysis_Global/1993.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13657 -O ERA5-Land_Reanalysis_Global/1993.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13658 -O ERA5-Land_Reanalysis_Global/1993.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13659 -O ERA5-Land_Reanalysis_Global/1993.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13660 -O ERA5-Land_Reanalysis_Global/1994.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13661 -O ERA5-Land_Reanalysis_Global/1994.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13662 -O ERA5-Land_Reanalysis_Global/1994.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13663 -O ERA5-Land_Reanalysis_Global/1994.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13664 -O ERA5-Land_Reanalysis_Global/1995.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13665 -O ERA5-Land_Reanalysis_Global/1995.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13666 -O ERA5-Land_Reanalysis_Global/1995.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13667 -O ERA5-Land_Reanalysis_Global/1995.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13668 -O ERA5-Land_Reanalysis_Global/1996.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13669 -O ERA5-Land_Reanalysis_Global/1996.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13670 -O ERA5-Land_Reanalysis_Global/1996.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13671 -O ERA5-Land_Reanalysis_Global/1996.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13672 -O ERA5-Land_Reanalysis_Global/1997.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13673 -O ERA5-Land_Reanalysis_Global/1997.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13674 -O ERA5-Land_Reanalysis_Global/1997.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13675 -O ERA5-Land_Reanalysis_Global/1997.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13676 -O ERA5-Land_Reanalysis_Global/1998.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13677 -O ERA5-Land_Reanalysis_Global/1998.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13678 -O ERA5-Land_Reanalysis_Global/1998.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13679 -O ERA5-Land_Reanalysis_Global/1998.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13680 -O ERA5-Land_Reanalysis_Global/1999.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13681 -O ERA5-Land_Reanalysis_Global/1999.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13682 -O ERA5-Land_Reanalysis_Global/1999.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13683 -O ERA5-Land_Reanalysis_Global/1999.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13684 -O ERA5-Land_Reanalysis_Global/2000.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13685 -O ERA5-Land_Reanalysis_Global/2000.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13686 -O ERA5-Land_Reanalysis_Global/2000.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13687 -O ERA5-Land_Reanalysis_Global/2000.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13688 -O ERA5-Land_Reanalysis_Global/2001.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13689 -O ERA5-Land_Reanalysis_Global/2001.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13690 -O ERA5-Land_Reanalysis_Global/2001.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13691 -O ERA5-Land_Reanalysis_Global/2001.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13692 -O ERA5-Land_Reanalysis_Global/2002.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13693 -O ERA5-Land_Reanalysis_Global/2002.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13694 -O ERA5-Land_Reanalysis_Global/2002.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13695 -O ERA5-Land_Reanalysis_Global/2002.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13696 -O ERA5-Land_Reanalysis_Global/2003.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13697 -O ERA5-Land_Reanalysis_Global/2003.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13698 -O ERA5-Land_Reanalysis_Global/2003.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13699 -O ERA5-Land_Reanalysis_Global/2003.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13700 -O ERA5-Land_Reanalysis_Global/2004.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13701 -O ERA5-Land_Reanalysis_Global/2004.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13702 -O ERA5-Land_Reanalysis_Global/2004.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13703 -O ERA5-Land_Reanalysis_Global/2004.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13704 -O ERA5-Land_Reanalysis_Global/2005.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13705 -O ERA5-Land_Reanalysis_Global/2005.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13706 -O ERA5-Land_Reanalysis_Global/2005.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13707 -O ERA5-Land_Reanalysis_Global/2005.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13708 -O ERA5-Land_Reanalysis_Global/2006.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13709 -O ERA5-Land_Reanalysis_Global/2006.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13710 -O ERA5-Land_Reanalysis_Global/2006.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13711 -O ERA5-Land_Reanalysis_Global/2006.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13712 -O ERA5-Land_Reanalysis_Global/2007.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13713 -O ERA5-Land_Reanalysis_Global/2007.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13714 -O ERA5-Land_Reanalysis_Global/2007.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13715 -O ERA5-Land_Reanalysis_Global/2007.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13716 -O ERA5-Land_Reanalysis_Global/2008.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13717 -O ERA5-Land_Reanalysis_Global/2008.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13718 -O ERA5-Land_Reanalysis_Global/2008.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13719 -O ERA5-Land_Reanalysis_Global/2008.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13720 -O ERA5-Land_Reanalysis_Global/2009.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13721 -O ERA5-Land_Reanalysis_Global/2009.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13722 -O ERA5-Land_Reanalysis_Global/2009.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13723 -O ERA5-Land_Reanalysis_Global/2009.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13724 -O ERA5-Land_Reanalysis_Global/2010.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13725 -O ERA5-Land_Reanalysis_Global/2010.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13726 -O ERA5-Land_Reanalysis_Global/2010.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13727 -O ERA5-Land_Reanalysis_Global/2010.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13728 -O ERA5-Land_Reanalysis_Global/2011.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13729 -O ERA5-Land_Reanalysis_Global/2011.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13730 -O ERA5-Land_Reanalysis_Global/2011.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13731 -O ERA5-Land_Reanalysis_Global/2011.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13732 -O ERA5-Land_Reanalysis_Global/2012.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13733 -O ERA5-Land_Reanalysis_Global/2012.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13734 -O ERA5-Land_Reanalysis_Global/2012.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13735 -O ERA5-Land_Reanalysis_Global/2012.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13736 -O ERA5-Land_Reanalysis_Global/2013.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13737 -O ERA5-Land_Reanalysis_Global/2013.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13738 -O ERA5-Land_Reanalysis_Global/2013.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13739 -O ERA5-Land_Reanalysis_Global/2013.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13740 -O ERA5-Land_Reanalysis_Global/2014.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13741 -O ERA5-Land_Reanalysis_Global/2014.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13742 -O ERA5-Land_Reanalysis_Global/2014.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13743 -O ERA5-Land_Reanalysis_Global/2014.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13744 -O ERA5-Land_Reanalysis_Global/2015.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13745 -O ERA5-Land_Reanalysis_Global/2015.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13746 -O ERA5-Land_Reanalysis_Global/2015.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13747 -O ERA5-Land_Reanalysis_Global/2015.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13748 -O ERA5-Land_Reanalysis_Global/2016.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13749 -O ERA5-Land_Reanalysis_Global/2016.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13750 -O ERA5-Land_Reanalysis_Global/2016.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13751 -O ERA5-Land_Reanalysis_Global/2016.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13752 -O ERA5-Land_Reanalysis_Global/2017.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13753 -O ERA5-Land_Reanalysis_Global/2017.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13754 -O ERA5-Land_Reanalysis_Global/2017.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13755 -O ERA5-Land_Reanalysis_Global/2017.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13756 -O ERA5-Land_Reanalysis_Global/2018.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13757 -O ERA5-Land_Reanalysis_Global/2018.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13758 -O ERA5-Land_Reanalysis_Global/2018.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13759 -O ERA5-Land_Reanalysis_Global/2018.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13760 -O ERA5-Land_Reanalysis_Global/2019.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13761 -O ERA5-Land_Reanalysis_Global/2019.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13762 -O ERA5-Land_Reanalysis_Global/2019.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13763 -O ERA5-Land_Reanalysis_Global/2019.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13764 -O ERA5-Land_Reanalysis_Global/2020.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13765 -O ERA5-Land_Reanalysis_Global/2020.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13766 -O ERA5-Land_Reanalysis_Global/2020.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13767 -O ERA5-Land_Reanalysis_Global/2020.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13768 -O ERA5-Land_Reanalysis_Global/2021.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13769 -O ERA5-Land_Reanalysis_Global/2021.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13770 -O ERA5-Land_Reanalysis_Global/2021.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13771 -O ERA5-Land_Reanalysis_Global/2021.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13772 -O ERA5-Land_Reanalysis_Global/2022.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13773 -O ERA5-Land_Reanalysis_Global/2022.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13774 -O ERA5-Land_Reanalysis_Global/2022.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13775 -O ERA5-Land_Reanalysis_Global/2022.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13776 -O ERA5-Land_Reanalysis_Global/2023.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13777 -O ERA5-Land_Reanalysis_Global/2023.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13778 -O ERA5-Land_Reanalysis_Global/2023.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13779 -O ERA5-Land_Reanalysis_Global/2023.7z.004

wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13780 -O ERA5-Land_Reanalysis_Global/2024.7z.001
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13781 -O ERA5-Land_Reanalysis_Global/2024.7z.002
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13782 -O ERA5-Land_Reanalysis_Global/2024.7z.003
wget --continue https://bonndata.uni-bonn.de/api/access/datafile/13783 -O ERA5-Land_Reanalysis_Global/2024.7z.004













