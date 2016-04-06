"""Thread utilities

"""
import os
import math

import urllib2 as urllib2
from requests.exceptions import HTTPError

from owslib.util import ServiceException

from omnivore.utils.background_http import BackgroundHttpMultiDownloader, BaseRequest, UnskippableRequest

from numpy_images import get_numpy_from_data

import rect

import logging
log = logging.getLogger(__name__)
#log.setLevel(logging.DEBUG)


loading_png = "\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x01\x00\x00\x00\x01\x00\x08\x06\x00\x00\x00\\r\xa8f\x00\x00\x00\x06bKGD\x00\xff\x00\xff\x00\xff\xa0\xbd\xa7\x93\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\x07tIME\x07\xdf\t\x16\x11/.\x84\xf0\x92\x8d\x00\x00\x00\x0ciTXtComment\x00\x00\x00\x00\x00\xbc\xae\xb2\x99\x00\x00\x05\xabIDATx\xda\xed\xdb\xcf\x8f$e\x1d\xc7\xf1OuMw\xcf\xec\xf4\xcc\xf4\xec\xec\xce\xc2.\x0bj\x00\x7fD\x88\xd1\xc4\x8b&\xc6x\x12\x8fz\xd4x\xf5\xe4A\x81\xf3\xdc\xbc{2\x1e \xe1\xc0A9\x021Q\x13\x13\x89\x91\xc4\xb0\xc6\xa8\xb0F\x84]2\x80;3.=S\xfd\xbbk\xca\x83\x0b\x89\x88\x7f\xc0\xf4\xbe^\xa7N\xa5\xfa\xf2M\x9ew\x9e\xa7\xaa\xbb\xd8\xdb\xdbk\x02\xdc\x9d\x04\x00\xee\xde\xb5\xdf2\x06\xb8{\t\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x02`\x04 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\xc0RZ1\x82\xe5\xf6\xd8\xb7\xbe\xfe\xaf\x9d\xdd\x9d\xed\xb2,\xff\xebz\xf1?w\x16\x1f\\+\x92,\xea\xd3\x1c\xde:\xbc\xfd\xc2s/\x9e7E\x01\xe0\x8c\xda\xd9\xdd\xd9~\xea\xc7O\xa7h\x15Y\x9c\xce?\xb4\xe0\x9b4w\x96\xfdh^%)\xd24\xad4\xa7M\x9a\xd4y\xfc\xc9\x1fn\x9b\xa0\x00p\x86\x95e\x99:\x8b\x14\xf3g3\xfc\xe5\xb5l\xffz\x94\x83'\xef\xcf\xfaF/\xd7\xaf\xff.\xaf\x0e\x7f\x9a\xb7\x0f\x06\xd9\xd8\xd8\xcc?\x8f\x06)W\xda9\x1eN\xf2\xdd\xc7\x9a\xb4WJ\x03\xf4\x0c\x80e0\x18\x1ce\xf0\xe9N\x8e\xab*\xa3_\xfc5U\xf5N\xaa\xaaJ1\xbd\x96\xe9\xf8$\xe3\xd1If\xe3*\xe3q\x95\xc5d\x98\xa28\xfd\xc8\x83\x02v\x00\x9c!\xef/\xe1\xad\xad\x8bI3\xcf\xb9\xef\x9fO~\xf4J\xba\xdf\xbc\x90\xdd{/gm\xfez\xc6\x8b\x87\xb2\xd1\xdfH\xa7le\xa5\xddI5\x9a\xa4(\xa6\x02`\x07\xc0\xb2d`0\x18\xa4\xaaN\xf2\xden\x9d\xe3\xedE\xc6?\xff{R\x9f\xa6\x19\xfd6\xf3i\x95\xd1\xc90\xf3\xe9(\xd3\xc90\xb3\xe9(E\xd1\x18\x9b\x1d\x00\xcbbkk7i\xe6\xe9omf\xf0\xbd\xcfd\xfb'GY|\xe5\xdet/Ls\xff\xec(\xed\xde\xd5\x9c\xeb\x96iw\xda9\x19\xcd\x92\x1c\xdb\x00\xd8\x01\xb0,\x87\x80\xc9d\x9aa5\xca\xe0x\x90\x93\xf5:\xb7\x1fN&\xcf\xfc#M=Ik\xf6b&\xe3*\xb3\xe90\x93a\x95\xd3\xd9\xf0\xce\xb7\x14\xc0\x0e\x80\xa5x\x06\xb0\xb6\xd6Oos\x98\xad\xcd\xcd$I\xfb\xdb\x1f\xcb\xda\x0f^\xcaJ\xfd@.\xef\xfc%\x93\xb5q\xae\xf4\xfe\x94n\xe7 \xb3\xf1\x9b)[\xdf\xb1\xfc\x05\x80e\t\xc0\xc9`\x98\xaa\x9a&\x19\xe6\x0f\xfb\xef\xe6\xe6\xbc\x95K\x0f\xd6\xf9\xe4\xd3\xd7\xb3\xff\x8da.\x95\xcfgz\xfc\x9b\xf4\xcf=\x9a\xee\xcbG)vF\x02\xe0\x08\xc0\xd9/\xc0\x7f~\xe1\xb7\xd9\xbf'Y\xed\xe7\x8f\xa3E\xde\x9c\xb5\xd3\xef_\xcc\xbb_{ \xf7\xadlg|p9\xfd\xdeq\xae\xdcX\xcd\xda\xaf^\xcb\xcdN\x9d\xa6\xdd1;;\x00\x96A\xd34y\xf9\xad7rm\xff\xb5\xf4z\x1b\x99L&\xa9\xaaa\xaa\xe94\xbf\xffT+\x8f\xfc\xec\xf5T\xaf\xdc\xc8\xfev;\xef|\xfe\x139n\x17\xb9\xda\xe9\x1a\x9c\x00\xb0\x0cG\x80\xd1b\x98\x83z\x9en\xa7\x97\xf5\xb5\xadL\xa6\xb3\xf4V{\xd9\xfa\xf3\x8d\xdc\xf7\xf6(\x9dvr\xfb\xb3\x8f\xe4\xad\x8fw\xd2\xed\xaef}6M\x1a\xaf\x01\x05\x80\xa5H@Q\xb4\xb2\x91\xcd\x8c\x8bQ\xba\xf5Z\xce\x15\xbd\\z\xf5V:\xfb\xe3\xbc\xf1\xa5G3}\xb0\xca\x97_z/\x7f\xbbzO\xcan;\xc9i\xea\xf9\xc23\x00\x01\xe0\xac[,\xea\xd4u\x9d+\xe7?\x97+;\xf9\xe0\xf5^\xf1P\x92\xaf&\x17\x8a;W\xbex\x9a/\x94\xf3\x94\xab\x9d\x94\xad2\xf5i\x9dE]\x1b\xa0\x00p\x96\x1d\xde:<|\xe2\x89\xc7/\x94\xed\x95|\xf8/\xbf\x1fu\\x\xff\xc3bQg}x\xd3\x00\x05\x80\xb3\xec\xf9\xe7^\xb8h\n\xfc?^\x03\x82\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00 \x00F\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00\xb0\\\x8a\xbd\xbd\xbd\xc6\x18\xe0\xee\xf4o\xa5'H\x03\xf5\xbd\ny\x00\x00\x00\x00IEND\xaeB`\x82"

error_png = '\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x01\x00\x00\x00\x01\x00\x08\x06\x00\x00\x00\\r\xa8f\x00\x00\x00\x06bKGD\x00\xff\x00\xff\x00\xff\xa0\xbd\xa7\x93\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\x07tIME\x07\xdf\t\x16\x11/\x0cQ\x90\xd3i\x00\x00\x00\x0ciTXtComment\x00\x00\x00\x00\x00\xbc\xae\xb2\x99\x00\x00\x05\xeaIDATx\xda\xed\xdb\xcf\x8b\x9cw\x01\xc7\xf1\xcf3\xb3?\xdd\x99\x9dI6\xbb\xcd&ms\xc8/\xaaB\xda\x93\x04zJ@\xda\x82PP\xc4\x83\x12\xf0R\xf1\xe0\xc5\xb4\'\xc1\xf5\xa4\xe0\xc9"=\xb4\x12A\xc4\x8bETZ\t\x84\xf4R\x10\xa4\xa5?\x90V\x12b\x9bj\x93\xe6\xc7\xc6dv\x9f\x99\x9d\xdd\x9d\x99\xc7\x83\xad\xd0\xea?\xb0\xb3\xaf\xd7\xf1a\xe7\xf2\x81\xef\x9b\xe7yf\xb6XYY\xa9\x02\xecN\x02\x00\xbb\xf7\xec\xd7\xcc\x00\xbb\x97\x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00 \x00&\x00\x01\x00\x04\x00\x10\x00@\x00\x00\x01\x00\x04\x00\x10\x00@\x00\x00\x01\x00\x04\x00\x10\x00@\x00\x00\x01\x00\x04\x00\x10\x00@\x00\x00\x01\x00\x04\x00\x10\x00@\x00\x00\x01\x00\x04\x00\x10\x00@\x00\x00\x01\x00\x04\x00\x10\x00@\x00\x00\x01\x00\x04\x00\x10\x00@\x00\x00\x01\x00\x04\x00\x10\x00@\x00@\x00\x00\x01\x00\x04\x00\x10\x00@\x00\x00\x01\x00\x04\x00\x10\x00@\x00\x00\x01\x00\x04\x00\x10\x00@\x00\x00\x01\x00\x04\x00\x10\x00@\x00\x00\x01\x00\x04\x00\x10\x00@\x00\x00\x01\x00\x04\x00\x10\x00@\x00\x00\x01\x00\x04\x00\x10\x00@\x00\x00\x01\x00\x04\x00\x10\x00@\x00\x00\x01\x00\x04\x00\x10\x00@\x00@\x00\x00\x01\x00\x04\x00\x10\x00@\x00\x00\x01\x00\xc6\xd2\x84\t\xc6\xdb\x13_{\xfc\xe2\xc2\xd2\xc2\xa9z\xbd\xfe\xa9\xeb\xc5\xff\xfce\xf1\xdfkE\x92\xc1p\x94\xd5[\xab\xaf\xbc\xfc\xe2\x9fN[Q\x00\xd8\xa1\x16\x96\x16N\x9d{\xf6\x97)jE\x06\xa3\xed\xcf\x1c\xf8*\xd5\xc7\xc7\xbe\xb7]&)RU\xb5T\xa3*U\x869\xfb\xcc\xf7OYP\x00\xd8\xc1\xea\xf5z\x86\x19\xa4\xd8\xfeM\xba\x17\xde\xcc\x9e\x8b\xbd\xdc~\xe6\xc1\xcc5\x1b\xb9t\xe9\xcf\xf9[\xf7\xf9\\\xbf\xddI\xb39\x9f\x9bw:\xa9OLf\xad\xdb\xcf\x99\'\xaaLN\xd4\r\xe8\x1d\x00\xe3\xa0\xd3\xb9\x93\xceCSY+\xcb\xf4\xce\xbf\x9b\xb2\xfc(eY\xa6\xd8|3\x9b\x1b\xeb\xd9\xe8\xadgk\xa3\xcc\xc6F\x99A\xbf\x9b\xa2\x18\xfd\xdf\x07\x05\xdc\x01\xb0\x83|r\x84[\xad\xc5\xa4\xda\xce\xe7\xbe\xb77\xf9\xf1\x1b\x99\xfe\xea\xbe,-\x1f\xc8\xec\xf6\xdf\xb318\x9af\xbb\x99\xa9z-\x13\x93S){\xfd\x14\xc5\xa6\x00\xb8\x03`\\2\xd0\xe9tR\x96\xeb\xb9\xb74\xcc\xda\x9eA6~{%\x19\x8eR\xf5^\xcd\xf6f\x99\xdez7\xdb\x9b\xbdl\xf6\xbb\xd9\xda\xec\xa5(*\xb3\t\x00\xe3\xa2\xd5ZJ\xa3\xb1\x90vk\x7ff\xbf\xf3\xf9\xecy\xa3\x96\x85\xc9\xe5,\xee[\xcc\x03\xfbVs\xe8\xc6\xf39\xb0\xd8\xca\x03\xf7\xb5rp\xa9\xfd\xe9\xdb\x07\x04\x80\x9d\xfd\x10\xd0\xefo\xa6[\xf6\xd2Y\xebd}n\x98\xbb\xc7\x92\xfe\xaf\xdeK5\xecg\xea\xda/r\xe2\xe4\x89\xcc\xbc\xff\xeb\xf4\xbbeF[\xdd\x8f?\xa5\x00\x02\xc0X\xbc\x03\x98\x9dm\xa71\xbf\x90Vk\x7f\x9a\x8d\xc5L~\xf3\xe1\xcc~0\x91|4\x93\x99\xcbog~\xf9\x91\xec\xbdw>\xc7\xab\x9f\xe7\xc4\x81s\xa9\xd7F\x8e\xff.\xe0%\xe0.\t\xc0z\xa7\x9b\xb2\xdcL\xd2\xcd\xeb\xd7n\xe4\x1f\xdb\xb5\xdcwd\x98\x03\x17\xde\xce\x17\xcf\xfc$\xc3\xb5?\xe6\x0b_9\x9b+\xaf\xfc,{\xde{0\xc5BO\x00\xdc\x01\xb0\xf3\x0b\xf0\x9f_\xf8\xcd\xb7\xf7\'3\xed\xbc\xd5\x1b\xe4\xea\xd6d\xda\xed\xc5\x14\x07\xcb\x0c&\xd6\xd2ho\xe4\x85\x9f\xfe!\x8d\xf6F\xb6o\xdc\xca\xb5\xf5K\xa9&\xa7l\'\x00\x8c\x83\xaa\xaa\xf2\x97\x7f\xbe\x9f\x97\xae\xbc\x9b\x0f\xffu\'\xfd~?\xbd{\x9d\xcc\xdd\xba\x99\xa3_\xffA\x86\xe5kI\x92a\xf9Z\x0e\x7fc%\xc3A/\xa3bd8\x01`\x1c\x1e\x01z\x83nn\x0f\xb73=\xd5\xc8\xdcl+3\xd3s9\xf8\xd7\xb7r\xf4\xe4\x994\xe7?\xc8h\xebz\x92d\xb4u=\xed\xb9\xab9r\xf2L\xee]x\xcex\x02\xc08$\xa0(jif>\x8d\xa2\x99\xe9\xe1l\xf6\xde\xede\xfa\xeez\x16\x8f\x1d\xca\xa8\x7f9\xc9(\xdf\xfe\xee\xfdIF\x19\xf5/g\xf1\xd8\xa1T\x83\xad\x1c\x9d\xb9i>\x01`\'\x1b\x0c\x86\x19\x0e\x879\xb8\xf7\xe1<r\xe4\xb1\x1c_~4\xcb\xaf_\xcc\xe1G\x9f\xcad\xe3pj\x8d\xd3\xa9\xb7\x9e\xcc\xb9\xe7>L\xbd\xf5dj\x8d\xd3\x99l\x1c\xce\xfd\xc7\xbf\x9c/5\xae\x1ap\xcc\xf9\x16`\xcc\xad\xdeZ=\xff\xf4\xd3g\x1f\xabON\xe4\x93\x7f\xf9\xfd\xd6\xbe\xe4\x9d\xdf\xfd(\xef\xfc\xbe\x96\xa2V\xa4\xa8\xd5\xf2P\xb3\xc8\xab\xcf\xfe0\xd5h\x94jT%\xa3Qf\x9b3\x06\x14\x00v\xb2\x97^|\xf9\xf1\xcf^{\xc1,x\x04\x00\x04\x00\x04\x00\x10\x00@\x00\x00\x01\x00\x04\x00\x10\x00@\x00\x00\x01\x00\x04\x00\x10\x00@\x00\x00\x01\x00\x04\x00\x10\x00@\x00\x00\x01\x00\x04\x00\x10\x00@\x00\x00\x01\x00\x04\x00\x10\x00@\x00\x00\x01\x00\x04\x00\x10\x00@\x00\x00\x01\x00\x04\x00\x10\x00@\x00\x00\x01\x00\x04\x00\x10\x00@\x00\x00\x01\x00\x010\x01\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x02\x00\x08\x00 \x00\x80\x00\x00\x02\x00\x08\x000\x96\x8a\x95\x95\x95\xca\x0c\xb0;\xfd\x1b\xa6\x88N\xb9\xbd\x88*T\x00\x00\x00\x00IEND\xaeB`\x82'



class TileServerInitRequest(UnskippableRequest):
    def __init__(self, tile_host, cache_root):
        self.tile_host = tile_host
        self.cache_root = cache_root
        UnskippableRequest.__init__(self)
        self.current_layer = None
        self.layer_keys = []
        self.world_bbox_rect = None
        
    def get_data_from_server(self):
        self.layer_keys = [self.tile_host.name]
        self.current_layer = self.tile_host.name
    
    def is_valid(self):
        return self.current_layer is not None
    
    def get_layer_info(self):
        layer_info = []
        for name in self.layer_keys:
            layer_info.append((name, self.tile_server.convert_title(self.tile_server[name].title)))
        return layer_info
    
    def get_default_layers(self):
        return list(self.layer_keys)
    
    def get_image(self, zoom, x, y):
        log.debug("TileServerInitRequest.get_image(%s,%s,%s)" % (zoom, x, y))
        data = loading_png
        return data
    
    def try_cache(self, zoom, x, y):
        print self.cache_root
        data = None
        if self.cache_root is not None:
            filename = self.tile_host.get_tile_cache_file(zoom, x, y)
            path = os.path.join(self.cache_root, filename)
            try:
                with open(path, "rb") as fh:
                    data = fh.read()
                # update last modified time so we can implement cache pruning
                # at some later time.
                os.utime(path, None)
                print "Found %s" % path
            except Exception, e:
                print "Failed reading %s" % path
                print "  exception:", e
        return data
    
    def save_cache(self, data, zoom, x, y):
        if self.cache_root is not None:
            filename = self.tile_host.get_tile_cache_file(zoom, x, y)
            path = os.path.join(self.cache_root, filename)
            try:
                dirname = os.path.dirname(path)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                with open(path, "wb") as fh:
                    fh.write(data)
            except:
                print "Failed caching %s" % path


class URLTileServerInitRequest(TileServerInitRequest):
    def get_image(self, zoom, x, y):
        if self.is_valid():
            data = self.try_cache(zoom, x, y)
            if data is None:
                url = self.tile_host.get_tile_url(zoom, x, y)
                print "requesting tile from %s" % url
                request = urllib2.Request(url)
                response = urllib2.urlopen(request)
                data = response.read()
                self.save_cache(data, zoom, x, y)
        else:
            data = error_png
        return data


class WMTSTileServerInitRequest(TileServerInitRequest):
    def get_data_from_server(self):
#        if True:  # To test error handling, uncomment this
#            import time
#            time.sleep(1)
#            self.error = "Test error"
#            return
        try:
            tile_server = WebMapService(self.url, self.tile_host.version)
            self.setup(tile_server)
        except ServiceException, e:
            self.error = e
        except HTTPError, e:
            log.error("Error %s contacting %s" % (e, self.url))
            self.error = e
        except AttributeError, e:
            log.error("Bad response from server" % self.url)
            self.error = e
        except Exception, e:
            log.error("Server error" % self.url)
            self.error = e
    
    def setup(self, tile_server):
        self.tile_server = tile_server
        self.layer_keys = self.tile_server.contents.keys()
        self.layer_keys.sort()
        self.current_layer = self.layer_keys[0]
        self.world_bbox_rect = self.get_global_bbox()
        self.debug()
    
    def get_global_bbox(self):
        bbox = ((None, None), (None, None))
        for name in self.layer_keys:
            b = self.tile_server[name].boundingBoxWGS84
            print "layer", name, "bbox", b
            r = ((b[0], b[1]), (b[2], b[3]))
            bbox = rect.accumulate_rect(bbox, r)
        return bbox
    
    def debug(self):
        tile_server = self.tile_server
        print tile_server
        print "contents", tile_server.contents
        layer = self.current_layer
        print "layer index", layer
        print "title", tile_server[layer].title
        print "bounding box", tile_server[layer].boundingBoxWGS84
        print "crsoptions", tile_server[layer].crsOptions
        print "styles", tile_server[layer].styles
        #    {'pseudo_bright': {'title': 'Pseudo-color image (Uses IR and Visual bands,
        #    542 mapping), gamma 1.5'}, 'pseudo': {'title': '(default) Pseudo-color
        #    image, pan sharpened (Uses IR and Visual bands, 542 mapping), gamma 1.5'},
        #    'visual': {'title': 'Real-color image, pan sharpened (Uses the visual
        #    bands, 321 mapping), gamma 1.5'}, 'pseudo_low': {'title': 'Pseudo-color
        #    image, pan sharpened (Uses IR and Visual bands, 542 mapping)'},
        #    'visual_low': {'title': 'Real-color image, pan sharpened (Uses the visual
        #    bands, 321 mapping)'}, 'visual_bright': {'title': 'Real-color image (Uses
        #    the visual bands, 321 mapping), gamma 1.5'}}

        #Available methods, their URLs, and available formats::

        print [op.name for op in tile_server.operations]
        print tile_server.getOperationByName('GetMap').methods

        # The NOAA server returns a bad URL (not fully specified or maybe just old),
        # so replace it with the server URL used above.  This prevents patching the
        # tile_server code.
        all_methods = tile_server.getOperationByName('GetMap').methods
        for m in all_methods:
            if m['type'].lower() == 'get':
                m['url'] = self.url
                break
        print tile_server.getOperationByName('GetMap').methods
        print tile_server.getOperationByName('GetMap').formatOptions
        
        for name in self.layer_keys:
            print "layer:", name, "title", self.tile_server.convert_title(self.tile_server[name].title), "crsoptions", tile_server[layer].crsOptions
    
    def get_bbox(self, layers, wr, pr):
        types = [("102100", "p"),
                 ("102113", "p"),
                 ("3857", "p"),
                 ("900913", "p"),
                 ("4326", "w"),
                 ]
        # FIXME: only using the first layer for coord sys.  Can different
        # layers have different allowed coordinate systems?
        layer = layers[0]
        bbox = None
        c = {t.split(":",1)[1]: t for t in self.tile_server[layer].crsOptions}
        for crs, which in types:
            if crs in c:
                if which == "p":
                    bbox = (pr[0][0], pr[0][1], pr[1][0], pr[1][1])
                else:
                    bbox = (wr[0][0], wr[0][1], wr[1][0], wr[1][1])
                break
        if bbox is None:
            bbox = (wr[0][0], wr[0][1], wr[1][0], wr[1][1])
        return c[crs], bbox
    
    def get_image(self, wr, pr, size, layers=None):
        if layers is None:
            layers = self.get_default_layers()
        corrected = []
        for name in layers:
            if not name:
                name = self.current_layer
            corrected.append(name)
        if self.is_valid():
            crs, bbox = self.get_bbox(corrected, wr, pr)
            img = self.tile_server.getmap(layers=corrected,
#                             styles=styles,
                             srs=crs,
                             bbox=bbox,
                             size=size,
                             format='image/png',
                             transparent=True
                             )
            data = img.read()
        else:
            data = loading_png
        return data

class TileHost(object):
    def __init__(self, name, url_list, strip_prefix="", tile_size=256, suffix=".png"):
        self.name = name
        self.urls = []
        for url in url_list:
            if url.endswith("?"):
                url = url[:-1]
            self.urls.append(url)
        self.url_index = 0
        self.num_urls = len(self.urls)
        self.strip_prefix = strip_prefix
        self.strip_prefix_len = len(strip_prefix)
        self.tile_size = tile_size
        self.suffix = suffix
    
    def __hash__(self):
        return hash(self.urls[0])
    
    # Reference for tile number calculations:
    # http://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
    
    def world_to_tile_num(self, zoom, lon, lat, clamp=True):
        zoom = int(zoom)
        if zoom == 0:
            return (0, 0)
        lat_rad = lat * math.pi / 180.0
        n = 2 << (zoom - 1)
        xtile = int((lon + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
        if clamp:
            xtile = max(xtile, 0)
            ytile = max(ytile, 0)
            xtile = min(xtile, n - 1)
            ytile = min(ytile, n - 1)
            
        return (xtile, ytile)
    
    rad2deg = 180.0 / math.pi
    
    def tile_num_to_world_lb_rt(self, zoom, x, y):
        zoom = int(zoom)
        if zoom == 0:
            return ((-180.0, -85.0511287798066), (180.0, 85.0511287798066))
        n = 2 << (zoom - 1)
        lon1 = (x * 360.0 / n) - 180.0
        lon2 = ((x + 1) * 360.0 / n) - 180.0
        lat1 = math.atan(math.sinh(math.pi * (1.0 - (2.0 * (y + 1) / n)))) * self.rad2deg
        lat2 = math.atan(math.sinh(math.pi * (1.0 - (2.0 * y / n)))) * self.rad2deg
        return ((lon1, lat1), (lon2, lat2))
    
    def get_tile_init_request(self, cache_root):
        raise NotImplementedError
    
    def get_tile_url(self, zoom, x, y):
        # mostly round robin URL index.  If multiple threads hit this at the
        # same time the same URLs might be used in each thread, but not worth
        # thread locking
        self.url_index = (self.url_index + 1) % self.num_urls
        url = self.urls[self.url_index]
        return "%s/%s/%s/%s%s" % (url, zoom, x, y, self.suffix)
    
    def get_tile_cache_file(self, zoom, x, y):
        # >>> ".".join("http://a.tile.openstreetmap.org/".split("//")[1].split("/")[0].rsplit(".", 2)[-2:])
        # 'openstreetmap.org'
        domain = ".".join(self.urls[0].split("//")[1].split("/")[0].rsplit(".", 2)[-2:])
        name = domain + "--" + "".join(x for x in self.name if x.isalnum())
        path = "%s/%s/%s/%s.png" % (name, zoom, x, y)
        return path


class LocalTileHost(TileHost):
    def __init__(self, name, tile_size=256):
        TileHost.__init__(self, name, [""], tile_size=tile_size)
    
    def __hash__(self):
        return hash(self.name)

    def get_tile_init_request(self, cache_root):
        return TileServerInitRequest(self, cache_root)


class OpenTileHost(TileHost):
    def get_tile_init_request(self, cache_root):
        return URLTileServerInitRequest(self, cache_root)

class OpenTileHostYX(OpenTileHost):
    def get_tile_url(self, zoom, x, y):
        # mostly round robin URL index.  If multiple threads hit this at the
        # same time the same URLs might be used in each thread, but not worth
        # thread locking
        self.url_index = (self.url_index + 1) % self.num_urls
        url = self.urls[self.url_index]
        return "%s/%s/%s/%s%s" % (url, zoom, y, x, self.suffix)


class WMTSTileHost(TileHost):
    def get_tile_init_request(self, cache_root):
        return WMTSTileServerInitRequest(self, cache_root)


class BackgroundTileDownloader(BackgroundHttpMultiDownloader):
    cached_known_tile_server = None
    
    def __init__(self, tile_host, cache_root):
        self.tile_host = tile_host
        if not os.path.exists(cache_root):
            try:
                os.makedirs(cache_root)
            except os.error:
                cache_root = None
        self.cache_root = cache_root
        BackgroundHttpMultiDownloader.__init__(self)

    @classmethod
    def get_known_tile_server(cls):
        if cls.cached_known_tile_server is None:
            cls.cached_known_tile_server = [
#                LocalTileHost("Blank"),
                
                # ESRI services listed here: http://server.arcgisonline.com/ArcGIS/rest/services/
                OpenTileHostYX("ESRI Topographic", ["http://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/"], suffix=""),
                OpenTileHostYX("ESRI USA Topographic", ["http://server.arcgisonline.com/ArcGIS/rest/services/USA_Topo_Maps/MapServer/tile/"], suffix=""),
                OpenTileHostYX("ESRI Ocean Base", ["http://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/"], suffix=""),
                OpenTileHostYX("ESRI Ocean Reference", ["http://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Reference/MapServer/tile/"], suffix=""),
                OpenTileHostYX("ESRI Terrain Base", ["http://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/"], suffix=""),
                OpenTileHostYX("ESRI Satellite Imagery", ["http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/"], suffix=""),
                OpenTileHostYX("ESRI Street Map", ["http://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/"], suffix=""),
                OpenTileHostYX("ESRI NatGeo Topographic", ["http://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/"], suffix=""),
                OpenTileHostYX("ESRI Shaded Relief", ["http://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/tile/"], suffix=""),
                
                OpenTileHost("MapQuest", ["http://otile1.mqcdn.com/tiles/1.0.0/osm/", "http://otile2.mqcdn.com/tiles/1.0.0/osm/", "http://otile3.mqcdn.com/tiles/1.0.0/osm/", "http://otile4.mqcdn.com/tiles/1.0.0/osm/"]),
                OpenTileHost("MapQuest Satellite", ["http://otile1.mqcdn.com/tiles/1.0.0/sat/", "http://otile2.mqcdn.com/tiles/1.0.0/sat/", "http://otile3.mqcdn.com/tiles/1.0.0/sat/", "http://otile4.mqcdn.com/tiles/1.0.0/sat/"]),
                OpenTileHost("OpenStreetMap", ["http://a.tile.openstreetmap.org/", "http://b.tile.openstreetmap.org/", "http://c.tile.openstreetmap.org/"]),
                OpenTileHost("Navionics", ["http://backend.navionics.io/tile/"], suffix="?LAYERS=config_2_20.00_0&TRANSPARENT=FALSE&UGC=TRUE&navtoken=TmF2aW9uaWNzX2ludGVybmFscHVycG9zZV8wMDAwMSt3ZWJhcHAubmF2aW9uaWNzLmNvbQ%3D%3D"),
                ]
        return cls.cached_known_tile_server
    
    @classmethod
    def get_tile_server_by_name(cls, name):
        for h in cls.get_known_tile_server():
            if h.name == name:
                return h
        return None

    def get_server_config(self):
        self.tile_server = self.tile_host.get_tile_init_request(self.cache_root)
        self.send_request(self.tile_server)
    
    def get_server(self):
        return self.tile_server
    
    def is_valid(self):
        return self.tile_server.is_valid()
    
    def request_tile(self, zoom, x, y, event=None, event_data=None):
        req = TileRequest(self.tile_server, zoom, x, y, event, event_data)
        self.send_request(req)
        return req


class TileRequest(UnskippableRequest):
    def __init__(self, tile_server, zoom, x, y, manager=None, event_data=None):
        UnskippableRequest.__init__(self)
        self.tile_server = tile_server
        self.zoom = zoom
        self.x = x
        self.y = y
        self.world_lb_rt = self.tile_server.tile_host.tile_num_to_world_lb_rt(self.zoom, self.x, self.y)
        self.url = "tile (%s,%s,z=%s)@%s from %s" % (x, y, zoom, self.world_lb_rt, tile_server.url)
        self.manager = manager
        self.event_data = event_data
    
    def get_data_from_server(self):
        try:
            self.data = self.tile_server.get_image(self.zoom, self.x, self.y)
        except urllib2.URLError, e:
            self.error = e
        except ServiceException, e:
            self.error = e
        except Exception, e:
            log.error("Error %s loading %s" % (e, self.url))
            self.error = e
        if self.manager is not None:
            self.manager.threaded_image_loaded = (self.event_data, self)
    
    def get_image_array(self):
        try:
            return get_numpy_from_data(self.data)
        except (IOError, TypeError), e:
            print "error converting image: %s" % e
            # some TileServeres return HTML data instead of an image on an error
            # (usually see this when outside the bounding box)
            return get_numpy_from_data(error_png)


if __name__ == "__main__":
    import time
    
    h = BackgroundTileDownloader.get_tile_server_by_name("OpenStreetMap")
    downloader = BackgroundTileDownloader(h, "tile_servertest")

    test = downloader.request_tile(0, 0, 0)
    test = downloader.request_tile(1, 0, 0)
    test = downloader.request_tile(1, 1, 0)
    while True:
        if test.is_finished:
            break
        time.sleep(1)
        print "Waiting for test..."

    if test.error:
        print "Error!", test.error
    else:
        print "world bbox", downloader.tile_server.world_bbox_rect
        outfile = 'tile_servertest.png'
        out = open(outfile, 'wb')
        out.write(test.data)
        out.close()
        print "Generated image", outfile
            
    downloader = None
