# coding=utf8
import math
import bisect
import numpy as np

# Enthought library imports.
from traits.api import on_trait_change, Unicode, Str, Any

from ..library import rect
from ..library.svg_utils import SVGOverlay

from base import ScreenLayer

import logging
log = logging.getLogger(__name__)

class CompassRose(ScreenLayer):
    """Compass Rose layer
    
    Shows a compass rose or north-up arrow as a graphic overlay
    """
    name = Unicode("Compass Rose")
    
    type = Str("compass_rose")
    
    # Sample compass rose from ElfQrin - Own work, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=15810328
    svg_source = Str("""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Creator: CorelDRAW -->

<svg
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
   xml:space="preserve"
   width="90mm"
   height="90mm"
   style="shape-rendering:geometricPrecision; text-rendering:geometricPrecision; image-rendering:optimizeQuality; fill-rule:evenodd; clip-rule:evenodd"
   viewBox="0 0 90 90"
   id="svg3545"
   version="1.1"
   inkscape:version="0.91 r13725"
   sodipodi:docname="Compass_rose_en_04p.svg"><metadata
     id="metadata3577"><rdf:RDF><cc:Work
         rdf:about=""><dc:format>image/svg+xml</dc:format><dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" /></cc:Work></rdf:RDF></metadata><sodipodi:namedview
     pagecolor="#ffffff"
     bordercolor="#666666"
     borderopacity="1"
     objecttolerance="10"
     gridtolerance="10"
     guidetolerance="10"
     inkscape:pageopacity="0"
     inkscape:pageshadow="2"
     inkscape:window-width="640"
     inkscape:window-height="480"
     id="namedview3575"
     showgrid="false"
     inkscape:zoom="2.1793827"
     inkscape:cx="159.44882"
     inkscape:cy="159.44882"
     inkscape:current-layer="svg3545" /><defs
     id="defs3547"><style
       type="text/css"
       id="style3549"><![CDATA[
    .str1 {stroke:#999999;stroke-width:0.0762}
    .str0 {stroke:#B2B2B2;stroke-width:0.0762}
    .fil0 {fill:none}
    .fil1 {fill:#9999FF}
    .fil2 {fill:#E5E5E5}
    .fil3 {fill:black;fill-rule:nonzero}
   ]]></style></defs><g
     id="_125485320"><polygon
       style="fill:#9999ff;stroke:#999999;stroke-width:0.0762"
       points="45.5813,45.003 53.596,37.003 45.5813,10.003 "
       class="fil1 str1"
       id="_125488368" /><polygon
       style="fill:#e5e5e5;stroke:#999999;stroke-width:0.0762"
       points="45.5813,45.003 37.5666,37.003 45.5813,10.003 "
       class="fil2 str1"
       id="_125485752" /></g><g
     id="_125486304"><polygon
       style="fill:#9999ff;stroke:#999999;stroke-width:0.0762"
       points="45.5813,45.003 53.5813,53.0177 80.5813,45.003 "
       class="fil1 str1"
       id="_125486160" /><polygon
       style="fill:#e5e5e5;stroke:#999999;stroke-width:0.0762"
       points="45.5813,45.003 53.5813,36.9883 80.5813,45.003 "
       class="fil2 str1"
       id="_125485464" /></g><g
     id="_125488488"><polygon
       style="fill:#9999ff;stroke:#999999;stroke-width:0.0762"
       points="45.5813,45.003 37.5666,53.003 45.5813,80.003 "
       class="fil1 str1"
       id="_125486136" /><polygon
       style="fill:#e5e5e5;stroke:#999999;stroke-width:0.0762"
       points="45.5813,45.003 53.596,53.003 45.5813,80.003 "
       class="fil2 str1"
       id="_125485440" /></g><g
     id="_125485632"><polygon
       style="fill:#9999ff;stroke:#999999;stroke-width:0.0762"
       points="45.5813,45.003 37.5813,36.9883 10.5813,45.003 "
       class="fil1 str1"
       id="_125488056" /><polygon
       style="fill:#e5e5e5;stroke:#999999;stroke-width:0.0762"
       points="45.5813,45.003 37.5813,53.0177 10.5813,45.003 "
       class="fil2 str1"
       id="_125486208" /></g><path
     style="fill:#000000;fill-rule:nonzero;stroke:#999999;stroke-width:0.0762"
     inkscape:connector-curvature="0"
     id="path3567"
     d="m 45.5427,4.0255 0.0868,0.3619 c 0.0804,0.3313 0.1598,0.6202 0.2381,0.8689 0.0794,0.2476 0.2805,0.8213 0.6054,1.7208 0.0307,-0.1143 0.092,-0.3661 0.1852,-0.7567 0.1206,-0.509 0.2,-0.8276 0.2392,-0.9578 0.1153,-0.3831 0.2275,-0.6635 0.3344,-0.8434 0.1079,-0.18 0.2275,-0.3091 0.3598,-0.3874 0.1323,-0.0794 0.2826,-0.1185 0.4498,-0.1185 0.1344,0 0.2371,0.0264 0.3091,0.0772 0.0709,0.0519 0.1068,0.1143 0.1068,0.1863 0,0.0519 -0.019,0.0974 -0.0582,0.1365 -0.0381,0.0392 -0.0825,0.0593 -0.1344,0.0593 -0.037,0 -0.0804,-0.0106 -0.1301,-0.0318 C 48.0308,4.2996 47.9695,4.2763 47.9483,4.2699 47.9112,4.2615 47.871,4.2572 47.8276,4.2572 c -0.1016,0 -0.1883,0.035 -0.2603,0.1027 -0.1281,0.1196 -0.2699,0.4212 -0.4233,0.9027 -0.1546,0.4816 -0.3504,1.251 -0.5874,2.3093 L 46.4507,8.0344 C 46.2147,7.5158 46.056,7.1475 45.9734,6.9306 45.8613,6.6385 45.7343,6.2712 45.5924,5.8267 L 45.2506,4.7536 c -0.0413,0.2476 -0.1376,0.6667 -0.29,1.2541 -0.1513,0.5884 -0.2423,0.9271 -0.2709,1.016 -0.0953,0.2784 -0.1969,0.4921 -0.3059,0.6414 -0.1079,0.1492 -0.2265,0.2571 -0.3556,0.3259 -0.1302,0.0678 -0.272,0.1016 -0.4265,0.1016 -0.1947,0 -0.3535,-0.0434 -0.4773,-0.1301 -0.0952,-0.0635 -0.1429,-0.145 -0.1429,-0.2445 0,-0.0434 0.017,-0.0826 0.0529,-0.1164 0.035,-0.0339 0.0773,-0.0508 0.127,-0.0508 0.0635,0 0.1334,0.0338 0.2075,0.1016 0.1323,0.1227 0.2529,0.1831 0.363,0.1831 0.1175,0 0.2328,-0.053 0.3471,-0.1577 0.1694,-0.162 0.3323,-0.4996 0.49,-1.0139 0.1895,-0.6223 0.3726,-1.4118 0.5482,-2.3686 -0.0719,0 -0.2106,0.0201 -0.4148,0.0614 -0.2053,0.0413 -0.3757,0.0995 -0.5133,0.1757 -0.1376,0.0751 -0.2434,0.1736 -0.3165,0.2942 -0.073,0.1207 -0.11,0.2477 -0.11,0.38 0,0.0656 0.0095,0.1333 0.0275,0.201 0.0127,0.0455 0.0402,0.1016 0.0815,0.1673 0.0466,0.0772 0.0709,0.1344 0.0709,0.1746 0,0.0487 -0.018,0.091 -0.054,0.127 -0.037,0.0349 -0.0815,0.0518 -0.1354,0.0518 -0.0784,0 -0.145,-0.0402 -0.2011,-0.1217 -0.0561,-0.0815 -0.0836,-0.2063 -0.0836,-0.3736 0,-0.2772 0.0698,-0.5133 0.2085,-0.7069 0.1397,-0.1948 0.327,-0.3419 0.5598,-0.4403 0.1492,-0.0624 0.436,-0.1217 0.8626,-0.1799 0.1566,-0.0212 0.3048,-0.0476 0.4434,-0.0804 z"
     class="fil3 str1" /><path
     style="fill:#000000;fill-rule:nonzero;stroke:#999999;stroke-width:0.0762"
     inkscape:connector-curvature="0"
     id="path3569"
     d="m 86.5366,43.8272 -0.1651,0 c 0.017,-0.1048 0.0254,-0.1884 0.0254,-0.2508 0,-0.0826 -0.0349,-0.1366 -0.1037,-0.163 -0.0698,-0.0254 -0.3302,-0.0381 -0.7831,-0.0381 l -0.453,0.0031 -0.344,1.3706 0.6424,0 c 0.2022,0 0.3271,-0.0159 0.3736,-0.0455 0.0466,-0.0297 0.0794,-0.1048 0.1006,-0.2244 l 0.1767,0 c -0.0667,0.2498 -0.1249,0.5249 -0.1767,0.8245 l -0.1588,0 c 0.0106,-0.0974 0.0159,-0.1726 0.0159,-0.2265 0,-0.0223 -0.0148,-0.0508 -0.0434,-0.0836 -0.0148,-0.017 -0.0413,-0.0276 -0.0804,-0.0339 -0.0561,-0.0106 -0.1429,-0.0159 -0.2604,-0.0159 l -0.636,0 c -0.1842,0.7028 -0.3747,1.2541 -0.5737,1.6531 0.1927,-0.0603 0.3631,-0.0899 0.5123,-0.0899 0.2476,0 0.5514,0.0391 0.9112,0.1175 0.1905,0.0433 0.3196,0.0656 0.3874,0.0656 0.091,0 0.164,-0.0233 0.2201,-0.0688 0.0561,-0.0455 0.0836,-0.0974 0.0836,-0.1545 0,-0.0339 -0.0127,-0.0868 -0.0402,-0.162 -0.0159,-0.0518 -0.0244,-0.0994 -0.0244,-0.145 0,-0.0476 0.0191,-0.0899 0.0572,-0.128 0.0381,-0.0371 0.0836,-0.055 0.1376,-0.055 0.0645,0 0.1174,0.0243 0.1598,0.074 0.0423,0.0498 0.0635,0.1154 0.0635,0.199 0,0.1757 -0.0773,0.3291 -0.2307,0.4614 -0.1546,0.1323 -0.3567,0.199 -0.6096,0.199 -0.1323,0 -0.2995,-0.0116 -0.5017,-0.0349 -0.4445,-0.0498 -0.7842,-0.0741 -1.017,-0.0741 -0.3059,0 -0.6636,0.0243 -1.0732,0.0741 l 0.0497,-0.1704 c 0.2435,-0.0677 0.3874,-0.1185 0.4308,-0.1503 0.0434,-0.0317 0.0794,-0.0794 0.109,-0.1408 0.0984,-0.1968 0.2275,-0.5905 0.3852,-1.1811 0.1588,-0.5916 0.3196,-1.216 0.4826,-1.8732 -0.2476,0 -0.4826,0.0593 -0.7038,0.1767 -0.2212,0.1175 -0.3937,0.271 -0.5175,0.4604 -0.1238,0.1895 -0.1863,0.3884 -0.1863,0.5969 0,0.0889 0.0085,0.1662 0.0265,0.2328 0.0169,0.0657 0.0572,0.1588 0.1196,0.2784 0.0392,0.073 0.0582,0.1312 0.0582,0.1767 0,0.0455 -0.019,0.0868 -0.0571,0.1217 -0.0381,0.035 -0.0858,0.053 -0.1408,0.053 -0.0804,0 -0.145,-0.0413 -0.1926,-0.1218 -0.0699,-0.1174 -0.1048,-0.2772 -0.1048,-0.4804 0,-0.434 0.164,-0.8234 0.4921,-1.1684 0.3292,-0.3461 0.8075,-0.5186 1.4362,-0.5186 l 1.0636,0 c 0.3673,0 0.6392,-0.0233 0.8149,-0.0709 -0.0825,0.3429 -0.1354,0.5873 -0.1577,0.7313 z"
     class="fil3 str1" /><path
     style="fill:#000000;fill-rule:nonzero;stroke:#999999;stroke-width:0.0762"
     inkscape:connector-curvature="0"
     id="path3571"
     d="m 47.073,83.1129 -0.1619,0 c 0.0232,-0.217 0.0349,-0.381 0.0349,-0.4932 0,-0.1587 -0.0434,-0.2847 -0.1291,-0.3768 -0.0858,-0.092 -0.2022,-0.1375 -0.3493,-0.1375 -0.2265,0 -0.4254,0.0825 -0.5948,0.2466 -0.1693,0.164 -0.254,0.3481 -0.254,0.5535 0,0.109 0.0233,0.2085 0.0699,0.2974 0.0455,0.0889 0.1492,0.2127 0.308,0.3725 0.291,0.2847 0.4826,0.508 0.5736,0.6689 0.091,0.1619 0.1365,0.3185 0.1365,0.472 0,0.2254 -0.0688,0.4424 -0.2064,0.6508 -0.1376,0.2085 -0.3397,0.3811 -0.6075,0.5165 -0.2677,0.1355 -0.5503,0.2032 -0.8477,0.2032 -0.2942,0 -0.5419,-0.0772 -0.7429,-0.2296 -0.2011,-0.1535 -0.3027,-0.3144 -0.3027,-0.4837 0,-0.091 0.0264,-0.1662 0.0794,-0.2265 0.0529,-0.0603 0.1132,-0.0899 0.1809,-0.0899 0.0709,0 0.1291,0.0275 0.1736,0.0836 0.0296,0.0328 0.0582,0.1079 0.0878,0.2233 0.0328,0.1407 0.0731,0.2423 0.1207,0.3058 0.0476,0.0625 0.1132,0.1143 0.1968,0.1546 0.0836,0.0402 0.1747,0.0603 0.2741,0.0603 0.2953,0 0.5546,-0.0995 0.7758,-0.2995 0.2201,-0.199 0.3313,-0.4202 0.3313,-0.6615 0,-0.0825 -0.0138,-0.1651 -0.0424,-0.2466 -0.0275,-0.0815 -0.0783,-0.1746 -0.1534,-0.2773 -0.0508,-0.0709 -0.1662,-0.1947 -0.344,-0.3725 -0.1757,-0.1736 -0.2857,-0.2921 -0.3312,-0.3567 -0.0646,-0.0931 -0.1122,-0.1862 -0.1429,-0.2804 -0.0307,-0.0942 -0.0466,-0.1873 -0.0466,-0.2805 0,-0.1884 0.0635,-0.3767 0.1895,-0.5672 0.1259,-0.1905 0.2899,-0.3419 0.491,-0.4541 0.2022,-0.1132 0.4308,-0.1693 0.6869,-0.1693 0.128,0 0.2868,0.0201 0.4741,0.0593 0.108,0.0201 0.2011,0.0359 0.2794,0.0465 -0.0455,0.1344 -0.0804,0.2604 -0.1058,0.38 -0.0244,0.1185 -0.0582,0.3545 -0.1016,0.708 z"
     class="fil3 str1" /><path
     style="fill:#000000;fill-rule:nonzero;stroke:#999999;stroke-width:0.0762"
     inkscape:connector-curvature="0"
     id="path3573"
     d="M 7.1136,47.324 7.175,45.3269 C 7.1898,44.8845 7.2311,44.4696 7.2988,44.0802 7.1792,44.2273 7.0289,44.4463 6.8469,44.7374 6.3029,45.6126 5.7166,46.4328 5.089,47.2001 5.0435,46.925 5.0064,46.6032 4.9789,46.2349 4.9504,45.8677 4.9366,45.5449 4.9366,45.2676 c 0,-0.3905 0.037,-0.7017 0.1101,-0.9345 0.073,-0.2328 0.2127,-0.4868 0.4169,-0.7641 -0.3884,0.0434 -0.6974,0.1661 -0.9249,0.3693 -0.2286,0.2022 -0.3429,0.4096 -0.3429,0.6223 0,0.0794 0.0211,0.1588 0.0645,0.2392 0.0519,0.0974 0.0783,0.1619 0.0783,0.1958 0,0.0529 -0.018,0.0974 -0.0518,0.1334 C 4.2529,45.1639 4.2106,45.1808 4.1619,45.1808 4.0952,45.1808 4.0412,45.1512 4,45.0909 3.9449,45.0125 3.9164,44.912 3.9164,44.7871 c 0,-0.1873 0.0539,-0.3799 0.1629,-0.5768 0.108,-0.1958 0.253,-0.3577 0.4329,-0.4847 0.1799,-0.127 0.3842,-0.2201 0.6138,-0.2773 0.126,-0.0317 0.3376,-0.0635 0.635,-0.0963 0.1016,-0.0127 0.2022,-0.0264 0.3017,-0.0402 -0.4488,0.6287 -0.6731,1.2626 -0.6731,1.9039 0,0.2001 0.0148,0.6001 0.0465,1.1991 0.1217,-0.146 0.2318,-0.2878 0.3313,-0.4244 0.0688,-0.0952 0.2466,-0.3587 0.5334,-0.7905 0.4635,-0.7028 0.8096,-1.1981 1.0393,-1.4849 0.1608,-0.1989 0.3429,-0.3841 0.545,-0.5556 -0.1979,0.78 -0.2974,1.8987 -0.2974,3.3581 L 8.3656,45.5534 C 8.5751,45.293 8.7858,44.9893 8.9995,44.6432 9.2144,44.2971 9.3551,44.0209 9.4229,43.8135 9.4663,43.679 9.4885,43.552 9.4885,43.4325 9.4885,43.315 9.461,43.1912 9.4081,43.0599 9.3742,42.9774 9.3583,42.9181 9.3583,42.8811 c 0,-0.0561 0.0201,-0.1038 0.0603,-0.1419 0.0403,-0.0381 0.0911,-0.0571 0.1535,-0.0571 0.0783,0 0.1408,0.0349 0.1863,0.1058 0.0656,0.0953 0.0984,0.2212 0.0984,0.3779 0,0.2084 -0.0709,0.4836 -0.2149,0.8244 -0.1439,0.3418 -0.4624,0.8393 -0.9567,1.4944 -0.6371,0.8382 -1.161,1.4509 -1.5716,1.8394 z"
     class="fil3 str1" /></svg>""")

    svg = Any

    skip_on_insert = True
    
    x_offset = 10
    y_offset = 50
    
    def _svg_default(self):
        return SVGOverlay(self.svg_source)

    @on_trait_change('svg_source')
    def svg_changed(self):
        self.svg = self._svg_default()

    # def pre_render(self, renderer, world_rect, projected_rect, screen_rect, layer_visibility):
    #     print "creating compass rose"
    #     self.svg = SVGOverlay(self.svg_source)

    def render_screen(self, renderer, w_r, p_r, s_r, layer_visibility, picker):
        if picker.is_active:
            return
        log.log(5, "Rendering scale!!! pick=%s" % (picker))
        render_window = renderer.canvas

        w = s_r[1][0]
        h = s_r[1][1]
        r = rect.get_rect_of_points([(w - 250, h - 250), (w - 50, h - 50)])
        renderer.draw_screen_svg(r, self.svg)